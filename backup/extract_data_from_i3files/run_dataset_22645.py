import multiprocessing
import subprocess


def start_process(argument_dict):
    script_path = '/home/storage/hans/jax_reco_gupta_corrections3/extract_data_from_i3files/convert_i3_ftr_calibrate.py'
    cmd = ['python']
    cmd.append(f"{script_path}")

    for key in argument_dict.keys():
        val = argument_dict[key]
        cmd.append("--"+key)
        cmd.append(f"{val}")

    #cmd.append("--recompute_true_muon_energy")
    return subprocess.run(cmd, shell=False)


if __name__ == '__main__':
    count = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(processes=count)

    arguments = []
    indir = f"/home/fast_storage/i3/22645/0000000-0000999/"
    outdir = f"/home/fast_storage/i3/22645/ftr/"
    infile_base = "FinalLevel_NuMu_NuGenCCNC"
    did = 22645

    delta=1000
    for i in range(1):
        min_file = i * delta
        max_file = (i+1) * delta

        argument_dict = dict()
        argument_dict['indir'] = indir
        argument_dict['infile_base'] = infile_base
        argument_dict['dataset_id'] = did
        argument_dict['outdir'] = outdir
        argument_dict['file_index_start'] = min_file
        argument_dict['file_index_end'] = max_file
        arguments.append(argument_dict)

    # start processes
    pool.map(start_process, arguments)
