import optparse
import os
from multiprocessing import Pool
import pandas as pd

from asm_catalog import choose_blocks_by_fdr_bh
from asm_utils import format_float
from handle_snps import classify_as_imprinting


def iterate_over_files_with_pandas(file_name, base_name):
    expected_cols = ["chrom", "start", "let1", "let2", "u1", "x1", "m1", "u2", "x2", "m2", "p_val", "let1_is_u"]
    used_cols = expected_cols[0:2] + expected_cols[-2:]
    zipped = list(zip(expected_cols, range(0, len(expected_cols))))
    zipped = zipped[0:2] + zipped[-2:]
    col_indices = [idx for col_name, idx in zipped]
    cur_homog = pd.read_csv(file_name, sep="\t", usecols=col_indices, names=used_cols)
    cur_homog = cur_homog[cur_homog["p_val"] >= 0]
    cur_homog["name"] = base_name
    return cur_homog


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--indir',
                      default="/cs/zbio/jrosensk/atlas_blocks_hg19/homog/homog_aligned/all_snps")
    parser.add_option('--outdir',
                      default="/cs/zbio/jrosensk/atlas_blocks_hg19/homog/homog_aligned/all_snps/")
    parser.add_option('--num_threads',
                      default=1)
    options, arguments = parser.parse_args()
    input_dir = options.indir
    num_threads = int(options.num_threads)
    is_first = True
    file_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith("allele.homog.pval.gz"):
            full_file_name = os.path.join(input_dir, filename)
            base_name = ".".join(filename.split(".")[:-4])
            if is_first:
                first_file = full_file_name
                is_first = False
            # if filename.split(".")[0] not in target_file_names:
            #     file_name_list_target.append((full_file_name, 0.0))
            file_list.append((full_file_name, base_name))

    with Pool(num_threads) as p:
        all_dfs = p.starmap(iterate_over_files_with_pandas, file_list)
        p.close()
        p.join()

    res_df = pd.concat(all_dfs)
    p_val_list = res_df["p_val"]
    let1_is_u_list = res_df["let1_is_u"]
    base_name_list = res_df["name"]
    snp_list = res_df["chrom"] + ":" + res_df["start"].apply(str)
    del res_df
    let1_and_base_name_list = list(zip(let1_is_u_list, base_name_list))
    snp_and_let1_base_name = list(zip(snp_list, let1_and_base_name_list))
    p_val_and_other = list(zip(p_val_list, snp_and_let1_base_name))
    sorted_list = sorted(p_val_and_other, key=lambda item: item[0])
    all_p_vals, all_snps_and_sample_names = [list(t) for t in zip(*sorted_list)]
    selected_p_vals, all_snps_and_sample_names, original_selected_p_vals = choose_blocks_by_fdr_bh(all_p_vals, all_snps_and_sample_names,
                                                                         alpha=0.1)

    snp_dict = {}
    snp_dict_hist = {}
    for (p, (snp, (let1_us_u, sample_name)), original_p) in zip(selected_p_vals, all_snps_and_sample_names, original_selected_p_vals):
        if snp in snp_dict:
            cur_dict = snp_dict[snp]
            p_val_list = cur_dict["p_val_list"]
            original_p_val_list = cur_dict["original_p_val_list"]
            sample_name_list = cur_dict["sample_names_list"]
            let1_is_u_list = cur_dict["let1_is_u"]
        else:
            cur_dict = {}
            p_val_list = []
            original_p_val_list = []
            sample_name_list = []
            let1_is_u_list = []
            cur_dict["bimodal_count"] = 0
            cur_dict["p_val_list"] = []
            cur_dict["original_p_val_list"] = []
            cur_dict["sample_names_list"] = []
            cur_dict["let1_is_u"] = []
        if snp in snp_dict_hist:
            cur_count = snp_dict_hist[snp]
        else:
            cur_count = 0

        # if snp == "chr1:40025080":
        #     x = 0
        p_val_list.append(p)
        original_p_val_list.append(original_p)
        sample_name_list.append(sample_name)
        let1_is_u_list.append(let1_us_u)
        cur_count += 1
        cur_dict["p_val_list"] = p_val_list
        cur_dict["original_p_val_list"] = original_p_val_list
        cur_dict["let1_is_u"] = let1_is_u_list
        cur_dict["sample_names_list"] = sample_name_list
        snp_dict[snp] = cur_dict
        snp_dict_hist[snp] = cur_count

    res_list = []
    all_asm_list = []
    for snp, cur_dict in snp_dict.items():
        p_val_list = cur_dict["p_val_list"]
        sample_name_list = cur_dict["sample_names_list"]
        bimodal_count = cur_dict["bimodal_count"]
        original_p_val_list = cur_dict["original_p_val_list"]
        let1_is_u_list = cur_dict["let1_is_u"]
        num_times_asm = len(let1_is_u_list)
        num_times_asm_let_1_u = sum(let1_is_u_list)
        all_asm_list.append([snp, ",".join([format_float(p) for p in p_val_list]), ",".join([format_float(p) for p in original_p_val_list]), ",".join(sample_name_list), str(bimodal_count)])
        # if snp == "chr1:40025080":
        #     x = 0
        if classify_as_imprinting(num_times_asm, num_times_asm_let_1_u):
            res_list.append([snp, ",".join([format_float(p) for p in p_val_list]), ",".join([format_float(p) for p in original_p_val_list]), ",".join(sample_name_list), str(bimodal_count)])
    # snps_with_at_least2_and_at_least_1_dif_u = [snp for snp in snps_with_at_least2 if (
    #             0 < sum([info_list[0] for info_list in snp_dict[snp]]) < len(snp_dict[snp]))]
    with open(os.path.join(options.outdir, "all_snps_imp_asm_res_table_all.01_fdr.tsv"), 'w') as f_out:
        header = "\t".join(['chrom', 'start', 'end', 'corrected_p-vals', 'p-vals', 'tissue_names', 'bimodal_count'])
        f_out.write(f"{header}\n")
        for res_info in res_list:
            snp_toks = res_info[0].split(":")
            snp_bed_info = [snp_toks[0], snp_toks[1], str(int(snp_toks[1]) + 1)]
            final_res_list = "\t".join(snp_bed_info + res_info[1:])
            f_out.write(f"{final_res_list}\n")
    with open(
            os.path.join(options.outdir, "all_asm.01_fdr.tsv"),
            'w') as f_out:
        header = "\t".join(['chrom', 'start', 'end', 'corrected_p-vals', 'p-vals', 'tissue_names', 'bimodal_count'])
        f_out.write(f"{header}\n")
        for res_info in all_asm_list:
            snp_toks = res_info[0].split(":")
            snp_bed_info = [snp_toks[0], snp_toks[1], str(int(snp_toks[1]) + 1)]
            final_res_list = "\t".join(snp_bed_info + res_info[1:])
            f_out.write(f"{final_res_list}\n")
    a = 0
    x = 0