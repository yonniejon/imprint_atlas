import argparse
import subprocess

import segment_betas
from segmentor_client import CHUNK_MAX_SIZE, SegmentByChunks
from utils_wgbs import GenomeRefPaths, pat_segment_tool, eprint, IllegalArgumentError, add_multi_thread_args, \
    add_GR_args
from multiprocessing import Pool
import numpy as np
import pandas as pd
import sys
import os

class PatProcessorParams:
    skip = None
    nsites = None
    homog_read_cutoff = None
    min_u_threshold = None
    min_m_threshold = None
    max_u_threshold = None
    max_m_threshold = None
    min_sites_per_read = None
    window_size = None

    def __init__(self, skip, nsite, homog_read_cutoff=None,
                 min_u_threshold=None, min_m_threshold=None,
                 max_u_threshold=None, max_m_threshold=None,
                 min_sites_per_read=None, window_size=None):
        self.skip = skip
        self.nsite = nsite
        self.homog_read_cutoff = homog_read_cutoff
        self.min_u_threshold = min_u_threshold
        self.min_m_threshold = min_u_threshold
        self.max_u_threshold = max_u_threshold
        self.max_m_threshold = max_m_threshold
        self.min_sites_per_read = min_sites_per_read
        self.window_size = window_size


def np2pd(arr):
    return pd.DataFrame({'startCpG': arr[:-1], 'endCpG': arr[1:]})


def is_overlap(b1, b2):
    return np.sum(find_dups(b1, b2))


def find_dups(b1, b2):
    return pd.Series(np.concatenate([b1, b2])).duplicated(keep='last')


def merge2(b1, b2):
    nr_from_df1 = np.argmax(np.array(find_dups(b1, b2)))
    skip_from_df2 = np.searchsorted(b2, b1[nr_from_df1])
    return np.concatenate([b1[:nr_from_df1 + 1], b2[skip_from_df2 + 1:]]).copy()


def increase_patch(pre_size, maxval):
    if pre_size == maxval:
        return maxval + 1
    return int(min(pre_size * 2, maxval))


def bimodal_switch_process(in_files, chrom, pat_processor_params, tmp_path):
    skip = pat_processor_params.skip
    nsites = pat_processor_params.nsite
    assert nsites, 'trying to segment an empty interval'.format(skip, nsites)
    if nsites == 1:
        return np.array([skip, skip + 1])
    try:
        # start_time = time.time()
        new_file_list = []
        for in_file in in_files:
            cur_basename = os.path.basename(in_file)
            if cur_basename.endswith(".gz"):
                cur_basename = cur_basename[:-7]
            else:
                cur_basename = cur_basename[:-4]
            cur_basename = cur_basename + "_{}_{}.pat".format(chrom, skip)
            out_path = tmp_path + cur_basename
            check_empty_cmd = "tabix {} {}:{}-{} | head | wc -l".format(in_file, chrom, str(skip), str(skip + nsites),
                                                                        out_path)
            tabix_cmd = "tabix {} {}:{}-{} > {}".format(in_file, chrom, str(skip), str(skip + nsites), out_path)
            is_empty_out_str = subprocess.check_output(check_empty_cmd, shell=True).decode().split()
            num_lines_count = int(is_empty_out_str[-1])
            if num_lines_count > 0:
                p = subprocess.Popen(tabix_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, error = p.communicate()
                new_file_list.append(out_path)
        if len(new_file_list) > 0:
            data_files = ' '.join(new_file_list)
            cmd = '{} {} -s {} -n {}'.format(pat_segment_tool, data_files, skip,
                                                          nsites)
            if pat_processor_params.homog_read_cutoff:
                cmd = cmd + " -read_homog_cutoff {}".format(pat_processor_params.homog_read_cutoff)
            if pat_processor_params.min_u_threshold:
                cmd = cmd + " -min_u_threshold {}".format(pat_processor_params.min_u_threshold)
            if pat_processor_params.min_m_threshold:
                cmd = cmd + " -min_m_threshold {}".format(pat_processor_params.min_m_threshold)
            if pat_processor_params.max_u_threshold:
                cmd = cmd + " -max_u_threshold {}".format(pat_processor_params.max_u_threshold)
            if pat_processor_params.max_m_threshold:
                cmd = cmd + " -max_m_threshold {}".format(pat_processor_params.max_m_threshold)
            if pat_processor_params.min_sites_per_read:
                cmd = cmd + " -cpgs_per_read {}".format(pat_processor_params.min_sites_per_read)
            if pat_processor_params.window_size:
                cmd = cmd + " -window_size {}".format(pat_processor_params.window_size)
            print("Finding blocks in chr {} start site {}".format(chrom, skip))
            blocks_list = subprocess.check_output(cmd, shell=True).decode().split("\n")
            for data_file in new_file_list:
                try:
                    os.remove(data_file)
                except Exception as e:
                    eprint("failed to remove file in s={}, n={}".format(skip, nsites))
                    eprint(e)
            return blocks_list, chrom
        else:
            return "", chrom

    except Exception as e:
        eprint('Failed in s={}, n={}'.format(skip, nsites))
        raise e


def break_to_chunks(start, end, step):
    """
    Break range of sites to chunks of size 'step'. Examples:

    skip=500, nsites=1500, step=500
    output: sls: [500, 1000]
            nls: [500, 500]

    skip=500, nsites=1550, step=500
    output: sls: [500, 1000]
            nls: [500, 550]
    """
    nsites = end - start

    if nsites < step:
        return np.array([start]), np.array([nsites])

    sls = np.arange(start, start + nsites, step)
    nls = np.ones_like(sls) * step

    if len(nls) * step > nsites:  # there's a leftover
        # update the last step:
        nls[-1] = nsites % step

        # in case last step is small, merge is with the one before last
        if nls[-1] < step // 5:
            sls = sls[:-1]
            nls = np.concatenate([nls[:-2], [np.sum(nls[-2:])]])
    return sls, nls


def break_to_chunks_by_chrome_bimodal(args, in_files, num_threads, step_size, tmp_path):
    chrom_list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11",
                  "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21",
                  "chr22", "chrX", "chrY"]
    cpg_file_path = GenomeRefPaths(name=args.genome).dict_path
    # chrom_list = ["chr7"]
    chunks_list = []
    for chrom in chrom_list:
        tabix_head_cmd = "tabix {} {} | head -1".format(cpg_file_path, chrom)
        tabix_tail_cmd = "tabix {} {} | tail -1".format(cpg_file_path, chrom)
        try:
            head_str = subprocess.check_output(tabix_head_cmd, shell=True).decode().split()
            tail_str = subprocess.check_output(tabix_tail_cmd, shell=True).decode().split()
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            raise Exception(f"Failed on command {tabix_head_cmd}")

        start_ind = int(head_str[-1])
        end_ind = int(tail_str[-1])
        skip_list, nsites_list = break_to_chunks(start_ind, end_ind, step_size)
        params = [(in_files, chrom, PatProcessorParams(si, ni, args.homog_read_cutoff, args.min_u_threshold, args.min_m_threshold,
                                                       args.max_u_threshold, args.max_m_threshold,
                                                       args.min_sites_per_read, args.window_size), tmp_path)
                  for si, ni in zip(skip_list, nsites_list)]
        chunks_list = chunks_list + params

    try:
        p = Pool(num_threads)
        arr = p.starmap(bimodal_switch_process, chunks_list)
        p.close()
        p.join()
    except Exception as e:
        raise e


    arr = [x for x, chrom in arr if len(x) > 1]
    flat_list = [item for sublist in arr for item in
                 sublist if len(item) > 0]
    args.max_bp = None
    args.pcount = None
    print("Printing blocks")
    SegmentByChunks(args, [], None).dump_pat_blocks(flat_list)


def parse_data_input(args):
    if args.pat_file:
        segment_betas.validate_single_file(args.pat_file)  # validate_single_file(args.beta_file)
        betas = [args.pat_file]
    else:
        assert(len(args.pats) > 0)
        betas = []
        for file_name in args.pats:
            # segment_betas.validate_single_file(file_name)
            betas.append(file_name)
    segment_betas.validate_files_list(betas)
    return betas

def parse_args():
    parser = argparse.ArgumentParser()
    add_GR_args(parser)
    # betas_or_file = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--pats', nargs='+')
    parser.add_argument('--pat_file')
    parser.add_argument('-c', '--chunk_size', type=int, default=CHUNK_MAX_SIZE,
                        help='Chunk size. Default {} sites'.format(CHUNK_MAX_SIZE))
    parser.add_argument('-w', '--window_size', type=float, default=4,
                        help='Minimum window size. Default 50')
    parser.add_argument('--min_sites_per_read', type=int, default=None,
                        help='Minimum number of CpG sites required to include a read. All reads with a smaller number '
                             'of CpG Sites will be ignored. Default is 3')
    parser.add_argument('--min_u_threshold', type=float, default=0.2,
                        help='Threshold on the minimum u proportion of reads in window to be included in a block.'
                             ' Must be a number between 0 and 1. Default is 0.2')
    parser.add_argument('--min_m_threshold', type=float, default=0.2,
                        help='Threshold on the minimum m proportion of reads in window to be included in a block.'
                             ' Must be a number between 0 and 1. Default is 0.2')
    parser.add_argument('--max_u_threshold', type=float, default=1,
                        help='Threshold on the maximum u proportion of reads in window to be included in a block.'
                             ' Must be a number between 0 and 1. Default is 1')
    parser.add_argument('--max_m_threshold', type=float, default=1,
                        help='Threshold on the max m proportion of reads in window to be included in a block.'
                             ' Must be a number between 0 and 1. Default is 1')
    parser.add_argument('--homog_read_cutoff', type=float, default=None,
                        help='Threshold of beta value of a read in order to consider it homogeneous (either U or M).'
                             ' Must be a number between 0 and 1. Default is 0.75')
    parser.add_argument('--tmp_path', default="tmp_dir",
                        help='path for creating temporary files [tmp_dir]')
    parser.add_argument('-o', '--out_path', default=sys.stdout,#"/cs/cbio/jon/trial_blocks.tsv",
                        help='output path [stdout]')
    add_multi_thread_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    betas = parse_data_input(args)
    if not os.path.isdir(args.tmp_path):
        os.makedirs(args.tmp_path)
    break_to_chunks_by_chrome_bimodal(args, betas, args.threads, args.chunk_size, args.tmp_path)
    try:
        os.removedirs(args.tmp_path)
    except Exception as e:
        eprint(f"failed to remove tmp dir {args.tmp_path}")
        eprint(e)



if __name__ == '__main__':
    main()