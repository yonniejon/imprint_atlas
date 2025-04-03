//
// Created by jrosensk on 6/25/23.
//

#ifndef SNPCOUNT_SNPMETHHISTB_H
#define SNPCOUNT_SNPMETHHISTB_H




#include <iostream>
#include <ctime>
//#include <string>
#include <vector>
#include <algorithm>    // std::sort
#include <fstream>
#include <regex>
#include <unordered_map>
#include <sstream>      // std::stringstream
#include <iomanip>      // std::setprecision
//#include <string_view>
#include <array>        // std::array
#include <cstdio>
#include <memory>
#include <stdexcept>

#define MAX_PAT_LEN 300
#define MAX_READ_LEN 1000
#define EXTEND_REGION 10000

struct reads_stats {
    int nr_pairs = 0;
    int nr_empty = 0;
    int nr_short = 0;
    int nr_invalid = 0;
    int nr_bad_conv = 0;
};

struct ReadOrient { // OT or OB
    char ref_chr;
    char unmeth_seq_chr;
    int shift;
    int mbias_ind;
};

struct mbias_ss {
    int meth[MAX_READ_LEN] = {0};
    int unmeth[MAX_READ_LEN] = {0};
};

struct SnpObj {
    char snp_let1;
    char snp_let2;
    int list_index;
    int locus;
};

struct ReadSNP {
    int locus;
    char snp_let;
    int list_index;
};

struct MethylData {
    int countMethyl; int countUnmethyl;
    int originalIndex;
    std::string pattern;
    std::vector<ReadSNP> pair_snps;
};

struct ReadMethData {
    std::string pattern;
    std::vector<ReadSNP> read_snps;
    int read_first_ind = -1;
    std::string chrom;
};

class SnpMethHist {
    std::string cpg_dict_path;
    std::string fasta_path;
    std::string snp_path;
    std::string snp_blacklist_path;

    // Current chromosome
    std::string chr;
    std::string region;
    int offset = 0;
    unsigned long int fasta_start = 0;
    unsigned long int fasta_end = 0;
    //bool conv[1000000] = {false};
    bool* conv;

    ReadOrient OT{'C', 'T', 0, 0};
    ReadOrient OB{'G', 'A', 1, 1};

    std::string mbias_path;
    int max_sauce_clip = 0;
    int min_cpg = 0;
    int clip_size = 0;
    int qual_filter = 0;
    std::vector<float> range;
    int bsize = 0;  // position of the last CpG in the current chromosome
    std::unordered_map<int, int> dict;
    std::string genome_ref;
    reads_stats readsStats;
    int line_i = 0;
    clock_t tick = clock();
    bool is_paired_end = false;
    std::vector <std::string> dummy_tokens;
    std::vector<std::vector<int>> snp_hist;
    std::unordered_map<int, SnpObj> snp_dict;
    std::unordered_map<int, SnpObj> snp_index_dict;
    bool first_line(std::string &line);

    mbias_ss mbias_OT[2];
    mbias_ss mbias_OB[2];


    std::vector<long> fasta_index();

    ReadMethData compareSeqToRef(std::string &seq, std::string &ref, int start_locus, int samflag, std::string &meth_pattern);
    void print_stats_msg();
    void dump_mbias();
    void print_progress();
    void load_fasta(std::string &chrom, unsigned long int start_locus);
    int locus2CpGIndex(int locus);

    std::string clean_CIGAR(std::string seq, std::string CIGAR);
    ReadMethData samLineToPatVec(std::vector<std::string> tokens);
    void proc2lines(std::vector<std::string> &tokens1, std::vector<std::string> &tokens2);
    void proc1line(std::vector <std::string> &tokens1);
    void load_genome_ref();
    void initialize_patter(std::string &line_str);

public:
    SnpMethHist(std::string cpg_refpath, std::string genome, std::string snppath, std::string snp_blacklist_path,
                 int mc, int qaulityFilter, std::vector<float> ranges, int smart_clip):
            cpg_dict_path(cpg_refpath), fasta_path(genome), snp_path(snppath), snp_blacklist_path(snp_blacklist_path), min_cpg(mc),
            qual_filter(qaulityFilter), range(ranges), max_sauce_clip(smart_clip) {}

    void action(std::string filename);
    void proc_sam_in_stream(std::istream& in);

    ~SnpMethHist() {delete[] conv;}


    SnpObj locus2SNP(int locus);

    char getSeqSnpLetter(char snp_let1, char snp_let2, char seq_snp_val, bool bottom);

    MethylData merge_and_count_methyl_data(ReadMethData read_res1, ReadMethData read_res2);

    MethylData meth_pattern_count(std::string meth_pattern);

    void dump_snphist();

    SnpObj index2SNP(int index);

    bool locus2SNPexists(int locus);
};

std::vector<std::string> line2tokens(std::string &line);
void print_vec(std::vector<std::string> &vec);


#endif //SNPCOUNT_SNPMETHHISTB_H
