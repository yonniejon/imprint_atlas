from utils_wgbs import validate_single_file, PAT2BETA_TOOL, PAT2RHO_TOOL, delete_or_skip, splitextgz, IllegalArgumentError, \
    GenomeRefPaths, trim_to_uint8
import subprocess
import os.path as op
from multiprocessing import Pool
import numpy as np


class PatProcesserOutput:
    BETA = 'beta'
    LBETA = 'lbeta'
    PAT = 'pat'



def process_pat(pat_path, out_dir, args, output_type, force=True):
    validate_single_file(pat_path)

    if pat_path.endswith('.pat.gz'):
        cmd = 'gunzip -cd'
    elif pat_path.endswith('.pat'):
        cmd = 'cat'
    else:
        raise IllegalArgumentError('Invalid pat suffix: {}'.format(pat_path))

    if output_type == PatProcesserOutput.BETA:
        suff = '.beta'
        is_lbeta = False
        tool = PAT2BETA_TOOL
    elif output_type == PatProcesserOutput.LBETA:
        suff = '.lbeta'
        is_lbeta = True
        tool = PAT2BETA_TOOL
    elif output_type == PatProcesserOutput.RHO:
        suff = '.rho'
        is_lbeta = False
        tool = PAT2RHO_TOOL

    out_file = op.join(out_dir, splitextgz(op.basename(pat_path))[0] + suff)
    if not delete_or_skip(out_file, force):
        return

    if args.threads > 1 and pat_path.endswith('.pat.gz') and op.isfile(pat_path + '.csi'):
        arr = mult_pat2processor(tool, pat_path, args)
    else:
        if args.nr_sites:
            nr_sites = args.nr_sites
        else:
            nr_sites = GenomeRefPaths(args.genome).nr_sites
        cmd += ' {} | {} {} {}'.format(pat_path, tool, args.start, int(args.start) + int(nr_sites) + 1)
        x = subprocess.check_output(cmd, shell=True).decode()
        arr = np.fromstring(x, dtype=int, sep=' ').reshape((-1, 2))

    trim_to_uint8(arr, is_lbeta).tofile(out_file)
    return out_file


def mult_pat2processor(tool, pat_path, args):
    processes = []
    with Pool(args.threads) as p:
        ct = GenomeRefPaths(args.genome).get_chrom_cpg_size_table()
        x = np.cumsum([0] + list(ct['size'])) + 1
        chroms = ['chr1'] #list(ct['chr'])
        for i, chrom in enumerate(chroms):
            start = x[i]
            end = x[i + 1]
            params = (tool, pat_path, chrom, start, end)
            processes.append(p.apply_async(chr_thread, params))
        p.close()
        p.join()

    processed_files = [pr.get() for pr in processes]
    res = np.concatenate(processed_files, axis = 0)
    return res


def chr_thread(tool, pat, chrom, start, end):
    cmd = 'tabix {} {} | '.format(pat, chrom)
    cmd += '{} {} {}'.format(tool, start, end)
    x = subprocess.check_output(cmd, shell=True).decode()
    x = np.fromstring(x, dtype=int, sep=' ').reshape((-1, 2))
    return x