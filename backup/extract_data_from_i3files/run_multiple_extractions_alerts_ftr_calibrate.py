#!/usr/bin/env python

import multiprocessing
import subprocess

event_ids = []

# low energy
#event_ids = [12547,
#    53129,
#    29183,
#    17424,
#    10145,
#    47271,
#    39474,
#    34495,
#    49730,
#    48963,
#    59593,
#    35147,
#    54227,
#    8789,
#    51931,
#    5352,
#    49010,
#    56061,
#    37503
#]

# high energy
event_ids = ['1022', '10393', '10644', '10738', '11086', '11232', '13011',
       '13945', '14017', '14230', '15243', '16416', '16443', '1663',
       '1722', '17475', '18846', '19455', '20027', '21113', '21663',
       '22232', '22510', '22617', '23574', '23638', '23862', '24530',
       '24726', '25181', '25596', '25632', '27063', '27188', '27285',
       '28188', '28400', '29040', '29707', '3062', '31920', '31989',
       '32781', '32839', '33119', '33656', '34506', '35349', '37086',
       '37263', '37448', '37786', '37811', '39166', '39962', '40023',
       '41381', '41586', '42566', '42568', '42677', '43153', '43483',
       '4397', '44081', '48309', '48448', '48632', '49067', '50832',
       '51687', '51956', '54374', '55301', '55526', '55533', '56041',
       '5620', '56741', '56774', '57174', '57394', '57723', '59010',
       '59029', '59089', '59099', '59228', '62274', '62512', '63373',
       '65472', '6586', '8', '8604', '8674', '8840', '9410', '9419',
       '9505']

def start_process(argument_dict):
    script_path = '/home/storage/hans/jax_reco_gupta_corrections2/extract_data_from_i3files/convert_i3_ftr_realtime_alerts_calibrate.py'
    cmd = ['python']
    cmd.append(f"{script_path}")

    #indir = "/home/storage2/hans/i3files/alerts/bfrv2/i3_nominal/"
    #outdir = "/home/storage2/hans/i3files/alerts/bfrv2/nominal/"

    indir = "/home/storage2/hans/i3files/alerts/bfrv2_no_hole_ice_flat/i3_nominal/"
    outdir = "/home/storage2/hans/i3files/alerts/bfrv2_no_hole_ice_flat/ftr/charge/calibrated/"

    #indir = '/home/storage2/hans/i3files/lowE/bfrv2_no_holeice_flat/i3_nominal/'
    #outdir = '/home/storage2/hans/i3files/lowE/bfrv2_no_holeice_flat/ftr/charge/calibrated/'

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
        inf_base = f"event_{e_id}_N100"
        argument_dict = dict()
        argument_dict['infile_base'] = inf_base
        arguments.append(argument_dict)

    # start processes
    pool.map(start_process, arguments)
