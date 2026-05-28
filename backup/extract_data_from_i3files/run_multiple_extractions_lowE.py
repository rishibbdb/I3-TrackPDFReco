#!/usr/bin/env python

import multiprocessing
import subprocess

event_ids = [12547, 53129, 29183, 17424, 10145,
 				47271, 39474, 34495, 49730, 48963,
 				59593, 35147, 54227, 8789, 51931,
 				5352, 49010, 56061, 37503]

def start_process(argument_dict):
    #script_path = '/home/storage/hans/jax_reco_gupta_corrections2/extract_data_from_i3files/convert_i3_ftr_lowE.py'
    script_path = '/home/storage/hans/jax_reco_gupta_corrections2/extract_data_from_i3files/convert_i3_tfrecord_lowE.py'
    cmd = ['python']
    cmd.append(f"{script_path}")

    #indir = "/home/storage2/hans/i3files/lowE/i3/"
    #outdir = "/home/storage2/hans/i3files/lowE/tfrecord/"

    indir = "/home/storage2/hans/i3files/lowE/bfrv2_no_holeice_flat/i3_nominal/"
    outdir = "/home/storage2/hans/i3files/lowE/bfrv2_no_holeice_flat/tfrecord/"

    cmd.append("--indir")
    cmd.append(f"{indir}")
    cmd.append("--outdir")
    cmd.append(f"{outdir}")

    for key in argument_dict.keys():
        val = argument_dict[key]
        cmd.append("--"+key)
        cmd.append(f"{val}")

    cmd.append("--recompute_true_muon_energy")
    return subprocess.run(cmd, shell=False)


if __name__ == '__main__':
    count = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(processes=count)

    arguments = []

    for e_id in event_ids:
        inf_base = f"event_{e_id}"
        argument_dict = dict()
        argument_dict['infile_base'] = inf_base
        arguments.append(argument_dict)

    # start processes
    pool.map(start_process, arguments)
