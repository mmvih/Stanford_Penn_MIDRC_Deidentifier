import argparse
import numpy as np
import torch
from hide_in_plain_sight import hide_in_plain_sight
from deidentifier_model import deidentifier_model
from multiprocessing import Process
import time
import random
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Deidentify radiology reports")
    parser.add_argument(
        "--device_list",
        nargs="+",
        help="Devices to run the transformer(s) model on, can provide several spaced device names to scale. Must be one of cpu, mps, cuda, or cuda:device_number",
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the transformer",
    )
    parser.add_argument(
        "--num_cpu_processes",
        type=int,
        default=1,
        help="If running on cpu, can split the deidentification between several processes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference with the transformer model",
    )
    parser.add_argument(
        "--max_number_of_reports_to_be_processed_in_parallel",
        type=int,
        default=50000,
        help="Max number of reports to process in parallel, should be lowered if "
        + "encountering memory issues, can be augmented if lots of remaining memory",
    )
    parser.add_argument(
        "--input_file_path",
        type=str,
        help="Path to input file, must be .npy",
        required=True,
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        help="Path to output file, must be .npy",
        required=True,
    )
    parser.add_argument(
        "--hospital_list",
        nargs="+",
        help="Hospitals to be parsed based on lower-case matching",
        required=False,
    )
    parser.add_argument(
        "--vendor_list",
        nargs="+",
        help="Vendors to be parsed based on lower-case matching",
        required=False,
    )

    return parser.parse_args()


def check_args_validity(args):
    assert args.batch_size >= 0
    assert args.input_file_path[-4:] == ".npy"
    assert args.output_file_path[-4:] == ".npy"

    assert len(args.device_list) > 0
    for device in args.device_list:
        if device not in ["cpu", "mps", "cuda"]:
            assert device[:5] == "cuda:"
            assert device[5:].isnumeric()


def deidentifier_model_and_hide_in_plain_sight(
    file_seed, device, num_workers, batch_size, hospital_list, vendor_list
):
    deidentifier_model(
        file_seed, device, num_workers, batch_size, hospital_list, vendor_list
    )
    # hide_in_plain_sight(file_seed)


def generate_output_files(file_seed_list, output_file_path):
    original_reports = []
    predictions = {}

    for file_seed in file_seed_list:
        with open("original_reports" + file_seed + ".npy", "rb") as f:
            original_reports.append(np.load(f, allow_pickle=True))
        os.remove("original_reports" + file_seed + ".npy")

        with open("predictions" + file_seed + ".npy", "rb") as f:
            prediction_seed = np.load(f, allow_pickle=True).item()
            for p in prediction_seed:
                if p in predictions:
                    predictions[p].extend(prediction_seed[p])
                else:
                    predictions[p] = prediction_seed[p]
        os.remove("predictions" + file_seed + ".npy")


    original_reports = np.concatenate(original_reports)
    # predictions = np.concatenate(predictions)

    with open("original_reports.npy", "wb") as f:
        np.save(f, original_reports, allow_pickle=True)

    with open("predictions.npy", "wb") as f:
        np.save(f, predictions, allow_pickle=True)

def main(args):
    # Load the reports
    start = time.time()

    with open(args.input_file_path, "rb") as f:
        reports = np.load(f, allow_pickle=True)

    print("Processing", str(len(reports)), "reports")

    device_list = (
        args.device_list
        if (len(args.device_list) > 1 or args.device_list[0] != "cpu")
        else ["cpu" for _ in range(args.num_cpu_processes)]
    )

    device_list = [torch.device(device) for device in device_list]
    file_seed_list = []
    number_batches = (
        len(reports) // args.max_number_of_reports_to_be_processed_in_parallel + 1
    )

    for batch_index in range(number_batches):
        file_seed_list_local = []

        print(
            "Processed so far",
            args.max_number_of_reports_to_be_processed_in_parallel * batch_index,
            "reports",
        )

        number_reports_in_the_batch = (
            args.max_number_of_reports_to_be_processed_in_parallel
            if batch_index < number_batches - 1
            else (
                len(reports)
                - args.max_number_of_reports_to_be_processed_in_parallel * batch_index
            )
        )

        if number_reports_in_the_batch == 0:
            assert batch_index == number_batches - 1
            assert args.max_number_of_reports_to_be_processed_in_parallel == len(
                reports
            )
            continue

        number_reports_per_file = number_reports_in_the_batch // len(device_list)

        # Prepare files to be processed in each separate process
        for i, device in enumerate(device_list):
            file_seed = (
                device.type
                + "device"
                + str(i)
                + "device"
                + str(batch_index)
                + "batch_index"
                + str(random.randint(2394492340, 23944923402394492340))
            )

            if i + 1 < len(device_list) and number_reports_per_file == 0:
                continue

            file_seed_list.append(file_seed)
            file_seed_list_local.append(file_seed)

            with open(
                "original_reports" + file_seed + ".npy",
                "wb",
            ) as f:
                np.save(
                    f,
                    np.array(
                        reports[
                            args.max_number_of_reports_to_be_processed_in_parallel
                            * batch_index
                            + i
                            * number_reports_per_file : (
                                (
                                    args.max_number_of_reports_to_be_processed_in_parallel
                                    * batch_index
                                    + (i + 1) * number_reports_per_file
                                )
                                if i + 1 < len(device_list)
                                else (
                                    args.max_number_of_reports_to_be_processed_in_parallel
                                    * batch_index
                                    + number_reports_in_the_batch
                                )
                            )
                        ]
                    ).astype("object"),
                    allow_pickle=True,
                )

        processes = []

        for file_seed, device in zip(file_seed_list_local, device_list):
            p = Process(
                target=deidentifier_model_and_hide_in_plain_sight,
                args=(
                    file_seed,
                    device,
                    args.num_workers,
                    args.batch_size,
                    args.hospital_list if args.hospital_list is not None else [],
                    args.vendor_list if args.vendor_list is not None else [],
                ),
            )
            p.start()
            processes.append(p)
            # Running without parallelization
            # deidentifier_model_and_hide_in_plain_sight(
            # file_seed,
            # device,
            # args.num_workers,
            # args.batch_size,
            # args.hospital_list,
            # args.vendor_list,
            # )
            # pass

        for p in processes:
            p.join()

        time.sleep(2)  # giving time for the processes to be cleaned

    generate_output_files(file_seed_list, args.output_file_path)

    print("Ended execution, took ", time.time() - start, " seconds")


if __name__ == "__main__":
    args = parse_args()
    check_args_validity(args)
    main(args)
