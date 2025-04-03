//
// Created by jrosensk on 6/25/23.
//

#include <set>
#include "SnpMethHist.h"
char METH = 'C';
char UNMETH = 'T';
char UNKNOWN = '.';
std::string TAB = "\t";



/***************************************************************
 * *
 *   Print methods *
 * *
 ***************************************************************/

std::vector <std::string> line2tokens(std::string &line) {
    /** Break string line to words (a vector of string tokens) */
    std::vector <std::string> result;
    std::string cell;
    std::stringstream lineStream(line);
    while (getline(lineStream, cell, '\t')) {
    result.push_back(cell);
    }
    return result;
}

bool is_cpg(std::string &seq, int j, ReadOrient ro) {
    /** check if a given sequence is a CpG site */
    if (ro.shift == 0) {
        return (j < seq.size() - 1) && ((seq[j] == 'C') || (seq[j] == 'T')) && (seq[j + 1] == 'G');
    } else {
        return (j > 0) && ((seq[j] == 'G') || (seq[j] == 'A')) && (seq[j - 1] == 'C');
    }
}

void print_vec(std::vector <std::string> &vec) {
/** print a vector to stderr, tab separated */
    std::string sep = "";
    for (auto &j: vec) {
        std::cerr << sep << j;
        sep = TAB;
    }
    std::cerr << std::endl;
}


std::string addCommas(int num) {
/** convert integer to string with commas */
    auto s = std::to_string(num);
    int n = s.length() - 3;
    while (n > 0) {
        s.insert(n, ",");
        n -= 3;
    }
    return s;
}

/***************************************************************
 * *
 *   Load FASTA *
 * *
 ***************************************************************/

std::string exec(const char* cmd) {
/** Execute a command and load output to string */
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("[ SnpMethHist ] popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

unsigned long int extend_region(std::string &region, int extend) {
    /** extend the region by 'extend' bases on each side */
    // in case the region is not in the expected format, return the original region
    // if it contains no ":", return the original region
    if (region.find(":") == std::string::npos) {
        return 0;
    }
    unsigned long int start = 1;
    std::regex re("(.+):([0-9]+)-([0-9]+)");
    std::smatch match;
    if (std::regex_search(region, match, re)) {
        // update end
        unsigned long int end = std::stol(match[3]);
        end += extend;
        // update start
        start = std::stol(match[2]);
        if (extend >= start) { extend = start; }
        start = std::max((unsigned long int)1, start - extend);
        region = std::string(match[1]) + ":" + std::to_string(start) + "-" + std::to_string(end);
    }
    return start - 1;
}

void SnpMethHist::load_fasta(std::string &cur_chrom, unsigned long int start_locus) {
    // load the current chromosome sequence from the FASTA file

//    offset = extend_region(region, MAX_READ_LEN);
    unsigned long int start = 1;
    start = std::max((long int)1, (long int)start_locus - (long int)EXTEND_REGION);
    unsigned long int end = start_locus + EXTEND_REGION;
    std::string cur_region = cur_chrom + ":" + std::to_string(start) + "-" + std::to_string(end);
    std::string cmd = "samtools faidx " + fasta_path + " " + cur_region;
    std::string chr_region = exec(cmd.c_str());
    if (chr_region.length() == 0) {
        throw std::invalid_argument("[ patter ] Error: Unable to read reference path: " + fasta_path);
    }
    std::stringstream ss(chr_region);

    // concatenate the lines to one long string
    for (std::string line_str; std::getline(ss, line_str, '\n');) {
        if (line_str[0] == '>') { continue; }
        for (auto &c: line_str) c = (char) toupper(c);  // change all characters to Uppercase
        genome_ref += line_str;
    }
    if (genome_ref.empty()) {
        throw std::invalid_argument("[ patter ] Error: Unable to read reference path: " + fasta_path);
    }
    offset = start - 1;
    fasta_start = start;
    fasta_end = end;
}

void SnpMethHist::load_genome_ref() {
/** Load the genome reference, corresponding to chromosome */

// load the current chromosome sequence from the FASTA file
    std::string cmd = "tabix " + cpg_dict_path + " " + chr + " | cut -f2-3";
    std::string cur_chr = exec(cmd.c_str());
    if (cur_chr.length() == 0) {
// If the region was empty due to lack of CpGs in range, bam2pat.py would have catched that earlier.
        throw std::invalid_argument("[ SnpMethHist ] Error: Unable to read reference path: " + cpg_dict_path);
    }
    std::stringstream ss(cur_chr);

    std::vector<std::string> tokens;
    std::vector<int> loci;
    for (std::string line_str; std::getline(ss, line_str, '\n');) {
        tokens = line2tokens(line_str);
        int locus = stoi(tokens[0]);
        int cpg_ind = stoi(tokens[1]);
        dict.insert(std::make_pair(locus, cpg_ind));
//std::cerr << locus << "  " << cpg_ind << std::endl;
        loci.push_back(locus);
    }
    offset = loci.at(0);
//std::cerr << "offset: " << offset << std::endl;
    bsize = loci.at(loci.size() - 1);

    conv = new bool[bsize]();
    for (int locus: loci) {
        conv[locus] = true;
    }

    std::string snp_black_cmd = "tabix " + snp_blacklist_path + " " + chr  + " | cut -f2-3";
    std::string cur_chr_snps_black = exec(snp_black_cmd.c_str());
    std::stringstream ss_snps_black(cur_chr_snps_black);

    std::vector<std::string> snp_black_tokens;
    std::vector<int> snp_black_loci;
    for (std::string line_str; std::getline(ss_snps_black, line_str, '\n');) {
        snp_black_tokens = line2tokens(line_str);

        int locus1 = stoi(snp_black_tokens[0]);
        int locus2 = stoi(snp_black_tokens[1]);
        conv[locus1] = false;
        conv[locus1 - 1] = false;
    }

/** Load SNP file */
    std::string snp_cmd = "tabix " + snp_path + " " + chr + " | cut -f2-4";
    std::string cur_chr_snps = exec(snp_cmd.c_str());
    if (cur_chr_snps.length() == 0) {
// If the region was empty due to lack of CpGs in range, bam2pat.py would have catched that earlier.
        throw std::invalid_argument("[ SNPer ] Error: No snps in " + chr + " in file " + snp_path);
        std::cerr <<  "[ SNPer ] No snps in " + chr + " in file " + snp_path << std::endl;
        return;
    }
    std::stringstream ss_snps(cur_chr_snps);
    
    std::vector<std::string> snp_tokens;
    int index = 0;
    for (std::string line_str; std::getline(ss_snps, line_str, '\n');) {
        snp_tokens = line2tokens(line_str);
        int locus = stoi(snp_tokens[0]);
        char c_snp_let1 = snp_tokens[1][0];
        char c_snp_let2 = snp_tokens[2][0];
        SnpObj curSnp{};
        curSnp.snp_let1 = c_snp_let1;
        curSnp.snp_let2 = c_snp_let2;
        curSnp.locus = locus;
        curSnp.list_index = index;

// snp_hist is initialized in header like: std::vector<std::vector<int>> snp_hist;
        snp_hist.push_back(std::vector<int>(6, 0));
        snp_index_dict.insert(std::make_pair(index, curSnp));
        index++;
// std::unordered_map<int, SnpObj> snp_dict;
        snp_dict.insert(std::make_pair(locus, curSnp));
        
        
    }


    return;
}

std::string vectorToString(const std::vector<int>& numbers) {
    std::ostringstream oss;

    // Iterate over the vector
    for (const auto& num : numbers) {
        // Append each number to the string stream
        oss << num << '\t';
    }

    // Convert the string stream to a string
    std::string result = oss.str();

    // Remove the trailing tab character
    if (!result.empty()) {
        result.pop_back();
    }

    return result;
}

void SnpMethHist::dump_snphist() {
    int cur_index = 0;
    for (const std::vector<int>& x : snp_hist){
        SnpObj cur_snp = index2SNP(cur_index);
        std::string hist_string = vectorToString(x);
        std::string s1(1, cur_snp.snp_let1);
        std::string s2(1, cur_snp.snp_let2);
        std::cout << chr + TAB + std::to_string(cur_snp.locus) + TAB + s1 + TAB + s2 + TAB + hist_string + "\n";
        cur_index++;
    }
}



/***************************************************************
 * *
 * Process single / pair of reads  *
 * *
 ***************************************************************/

std::string SnpMethHist::clean_CIGAR(std::string seq, std::string CIGAR) {

/** use CIGAR string to adjust 'seq' so it will be comparable to the reference.
* e.g, remove false letters ('I'), insert fictive letters ('D') etc. */

// parse CIGAR and convert it to a couple of vectors: chars, nums.
// e.g, '2S9M' will become ['S', 'M'] and [2, 9]
    std::vector<char> chars;
    std::vector<unsigned long> nums;
    std::string cur_num;
    for (auto c: CIGAR) {
        if (isdigit(c)) {
            cur_num.push_back(c);
        } else {
            nums.push_back(stoul(cur_num));
            cur_num.clear();
            chars.push_back(c);
        }
    }

// build the adjusted seq
    std::string adjusted_seq;
    for (int i = 0; i < (int) chars.size(); i++) {
        char ch = chars[i]; // CIGAR character
        int num = nums[i];  // corresponding integer
        if (ch == 'M') {
            adjusted_seq += seq.substr(0, num);
            seq = seq.substr(num, seq.length() - num);
        } else if (ch == 'D') {
            for (unsigned long j = 0; j < num; j++)
                adjusted_seq += 'N';
        } else if ((ch == 'I') || (ch == 'S')) {
            seq = seq.substr(num, seq.length() - num);
        } else if ((ch == 'H')) {
            continue;
        } else {
            throw std::invalid_argument("[ SnpMethHist ] Unknown CIGAR character: " +
                                        std::string(1, ch));
        }
    }

    return adjusted_seq;
}


int SnpMethHist::locus2CpGIndex(int locus) {
/** translate genomic locus to CpG index (in range 1,...,28M~) */
    int start_site = 0;
    auto search = dict.find(locus);
    if (search != dict.end()) {
        start_site = search->second;
    } else {
// Should never happen. Probably means a reference mismatch
        throw std::logic_error("[ SnpMethHist ] Reference Error. Unknown CpG locus: " +
                               std::to_string(locus));
    }
    return start_site;
}

SnpObj SnpMethHist::locus2SNP(int locus) {
/** translate genomic locus to Snp object with snp letters */
    SnpObj c_snp;
    auto search = snp_dict.find(locus);
    if (search != snp_dict.end()) {
        c_snp = search->second;
    } else {
        throw std::logic_error("[ SnpMethHist ] Reference Error. Unknown SNP locus: " +
                               std::to_string(locus));
    }
    return c_snp;
}

bool SnpMethHist::locus2SNPexists(int locus) {
/** translate genomic locus to Snp object with snp letters */
    SnpObj c_snp;
    auto search = snp_dict.find(locus);
    if (search != snp_dict.end()) {
        c_snp = search->second;
    } else {
        return false;
    }
    return true;
}

SnpObj SnpMethHist::index2SNP(int index) {
/** translate genomic locus to Snp object with snp letters */
    SnpObj c_snp;
    auto search = snp_index_dict.find(index);
    if (search != snp_index_dict.end()) {
        c_snp = search->second;
    } else {
// Should never happen. Probably means a reference mismatch
        throw std::logic_error("[ SNPMethHist ] Reference Error. Unknown SNP index: " +
                               std::to_string(index));
    }
    return c_snp;
}


int strip_pat(std::string &pat) {
// remove dots from the tail (e.g. CCT.C.... -> CCT.C)
    pat = pat.substr(0, pat.find_last_not_of(UNKNOWN) + 1);
    if (pat == "") { return -1; }
// remove dots from the head (..CCT -> CCT)
    int pos = pat.find_first_not_of(UNKNOWN);
    if (pos > 0) {
        pat = pat.substr(pos, pat.length() - pos);
    }
    return pos;
}

char SnpMethHist::getSeqSnpLetter(char snp_let1, char snp_let2, char seq_snp_val, bool bottom) {
    char snp_letter = 'Z';
    if (((snp_let1 == 'C' && snp_let2 == 'T') || (snp_let2 == 'C' && snp_let1 == 'T')) && !bottom){
        return snp_letter;
    }
    if (((snp_let1 == 'G' && snp_let2 == 'A') || (snp_let2 == 'G' && snp_let1 == 'A')) && bottom){
        return snp_letter;
    }
    std::set<char> allowed_letters1;
    if (snp_let1 == 'C' && snp_let2 != 'T' && !bottom) {
        allowed_letters1 = {'C', 'T'};
    } else if (snp_let1 == 'G' && snp_let2 != 'A' && bottom) {
        allowed_letters1 = {'G', 'A'};
    } else {
        allowed_letters1 = {snp_let1};
    }
    std::set<char> allowed_letters2;
    if (snp_let2 == 'C' && snp_let1 != 'T' && !bottom) {
        allowed_letters2 = {'C', 'T'};
    } else if (snp_let2 == 'G' && snp_let1 != 'A' && bottom) {
        allowed_letters2 = {'G', 'A'};
    } else {
        allowed_letters2 = {snp_let2};
    }
    if (allowed_letters1.find(seq_snp_val) != allowed_letters1.end()){
        snp_letter = snp_let1;
    } else if (allowed_letters2.find(seq_snp_val) != allowed_letters2.end()) {
        snp_letter = snp_let2;
    }
    return snp_letter;
}

int get_sauce_clip_top(std::string &seq, std::string &ref, int max_sauce_clip){
    int len_seq = seq.length();
    int i = len_seq - 1;
    while(((len_seq - i) < max_sauce_clip) && i >= 0){
        if (((ref[i] == 'A') && (seq[i] == 'A')) || ((ref[i] == 'G') && (seq[i] == 'G'))){
            break;
        }
        i--;
    }
    while(i < len_seq){
        if ((ref[i] == 'C') && (seq[i] =='C')){
            break;
        }
        if((ref[i] != seq[i]) || (ref[i] == 'C' && seq[i] == 'T')){
            break;
        }
        i++;
    }
    return i;
}

int get_sauce_clip_bottom(std::string &seq, std::string &ref, int max_sauce_clip){
    int len_seq = seq.length();
    int i = 0;
    while((i < max_sauce_clip) && i < len_seq){
        if (((ref[i] == 'C') && (seq[i] == 'C')) || ((ref[i] == 'T') && (seq[i] == 'T'))){
            break;
        }
        i++;
    }
    while(i > 0){
        if ((ref[i] == 'G') && (seq[i] =='G')){
            break;
        }
        if(ref[i] != seq[i] || (ref[i] == 'G' && seq[i] == 'A')){
            break;
        }
        i--;
    }
    return i;
}

int get_sauce_clip(std::string &seq, std::string &ref, bool bottom, int max_sauce_clip) {
    if (bottom) {
        return get_sauce_clip_bottom(seq, ref, max_sauce_clip);
    } else {
        return get_sauce_clip_top(seq, ref, max_sauce_clip);
    }
}

ReadMethData SnpMethHist::compareSeqToRef(std::string &seq, std::string &ref,
                                           int start_locus,
                                           int samflag,
                                           std::string &meth_pattern) {
/** compare seq string to ref string. generate the methylation pattern, and return
* the CpG index of the first CpG site in the seq (or -1 if there is none) */

// get orientation
    bool bottom;
    bool top;
    if (is_paired_end) {
        bottom = ( ((samflag & 0x53) == 83) || ((samflag & 0xA3) == 163) );
        top = ( ((samflag & 0x63) == 99) || ((samflag & 0x93) == 147) );
    } else {
        bottom = ((samflag & 0x10) == 16);
    }
    if (max_sauce_clip > 0){
        int sauce_clip_index = get_sauce_clip(seq, ref, bottom, max_sauce_clip);
        //std::cerr << "flag: " << samflag << ". is bottom: " << bottom << ". sause clip size: " << sauce_clip_index << std::endl;
        if (bottom){
            seq = seq.substr(sauce_clip_index, seq.length());
            ref = ref.substr(sauce_clip_index, seq.length());
            start_locus = start_locus + sauce_clip_index;
        } else {
            seq = seq.substr(0, sauce_clip_index);
            ref = ref.substr(0, sauce_clip_index);
        }
    }
    bool either_bottom_or_top = false;
    if (!bottom && top){
        either_bottom_or_top = true;
    } else if (!top && bottom){
        either_bottom_or_top = true;
    }
    ReadOrient ro = bottom ? OB : OT;

    std::vector<ReadSNP> read_snps;

// get flag for mbias
    int mbias_ind;
    mbias_ss *mb;
    bool skip_mbias = true;

// generate the methylation pattern (e.g 'CC.TC'),
// by comparing the given sequence to reference at the CpG indexes
    char cur_status;
    int j;
    int mj;
    int first_ind = -1;
    for (unsigned long i = 0; i < seq.length(); i++) {

// this deals with the case where a read exceeds
// the last CpG of the chromosome. Ignore the rest of the read.
        if ((start_locus + i) > (bsize - 1)) {
            continue;
        }
        mj = bottom ? (seq.length() - i - 1) : i;
        if (mj >= MAX_READ_LEN) {skip_mbias = true;} // read is too long. skip it to avoid seg. fault
        if (locus2SNPexists(start_locus + i)) {
            if(!either_bottom_or_top){
                continue;
            }
            SnpObj curSnp = locus2SNP(start_locus + i);
            char snp_val = seq[i];
            char read_snp_let = getSeqSnpLetter(curSnp.snp_let1, curSnp.snp_let2, snp_val, bottom);
            if (read_snp_let == 'Z'){
                continue;
            }
            ReadSNP cur_snp;
            cur_snp.locus = start_locus + i;
            cur_snp.snp_let = read_snp_let;
            cur_snp.list_index = curSnp.list_index;
            read_snps.push_back(cur_snp);
        } else{
            if (conv[start_locus + i]) {
                j = i + ro.shift;  // add a 1-pos shift for bottom strands
                char s = seq[j];
                cur_status = UNKNOWN;
                // check if this is a CpG site. If not, skip it.
                if (!is_cpg(seq, j, ro)) {
                    cur_status = UNKNOWN;
                } else {
                    if (s == ro.unmeth_seq_chr) {
                        cur_status = UNMETH;
                    }
                    else if (s == ro.ref_chr) {
                        cur_status = METH;
                    }
                }

// ignore first/last 'clip_size' characters, since they are often biased
// TODO: allow 4 different clip sizes (OT,OB,CTOT,CTOB)
                if (!((j >= clip_size) && (j < seq.size() - clip_size))) {
                    cur_status = UNKNOWN;
                }
// find first CpG index
                if ((first_ind < 0) && (cur_status != UNKNOWN)) {
                    first_ind = locus2CpGIndex(start_locus + i);
                }
                if (first_ind > 0) {
                    meth_pattern.push_back(cur_status);
                }
            }
        }
    }
    ReadMethData read_meth_data;
    if (strip_pat(meth_pattern)) {return read_meth_data;}
    read_meth_data.pattern = meth_pattern;
    read_meth_data.read_snps = read_snps;
    read_meth_data.read_first_ind = first_ind;
    return read_meth_data;
}

ReadMethData merge(ReadMethData read_res1, ReadMethData read_res2) {
/** Merge 2 complementary lines to a single output.
* each line has the following fields: [chr, startCpG, pat]
* One or more of the lines may be empty */

// if one of the lines is empty - return the other
    if (read_res1.read_first_ind == -1) { return read_res2; }
    if (read_res2.read_first_ind == -1) { return read_res1; }

// Swap lines s.t l1 starts before l2
    if (read_res1.read_first_ind > read_res2.read_first_ind) {
        ReadMethData tmp = read_res1;
        read_res1 = read_res2;
        read_res2 = tmp;
    }

    int start1 = read_res1.read_first_ind, start2 = read_res2.read_first_ind;
    std::string pat1 = read_res1.pattern, pat2 = read_res2.pattern;

    std::string merged_pat;  // output pattern
    int last_site = std::max(start1 + pat1.length(), start2 + pat2.length()); // location of last CpG from both reads

    if (last_site - start1 > MAX_PAT_LEN) // sanity check: make sure the two reads are not too far apart
        throw std::invalid_argument("invalid pairing. merged read is too long");

// init merged_pat with missing values
    for (int i = start1; i < last_site; i++)
        merged_pat += ".";

// set merged_pat head with pat1
    for (unsigned long i = 0; i < pat1.length(); i++)
        merged_pat[i] = pat1[i];

// set pat2 in the adjusted position
    for (unsigned long i = 0; i < pat2.length(); i++) {
        int adj_i = i + start2 - start1;
        if (merged_pat[adj_i] == UNKNOWN) {   // this site was missing from read1
            merged_pat[adj_i] = pat2[i];
        } else if ((pat2[i] != UNKNOWN) && (merged_pat[adj_i] != pat2[i])) {
// read1 and read2 disagree, and none of them is missing ('.').
// treat this case as a missing value for now
// future work: consider only the read with the higher quality.
            merged_pat[adj_i] = UNKNOWN;
        }
    }
// strip merged pat (remove trailing dots):
    int pos = strip_pat(merged_pat);
    if (pos < 0 ) { return {}; }
    read_res1.read_first_ind = start1 + pos;
    read_res1.pattern = merged_pat;
    return read_res1;
}

MethylData SnpMethHist::merge_and_count_methyl_data(ReadMethData read_res1,
                                                     ReadMethData read_res2) {
/** Merge 2 complementary lines to a single output.
* each line has the following fields: [chr, startCpG, pat, start_bp, read_len_bp]
* One or more of the lines may be empty */


    if (read_res1.read_first_ind == -1 && read_res2.read_first_ind == -1){
        MethylData res;
        res.countMethyl = 0;
        res.countUnmethyl = 0;
        res.pattern = "";
        return res;
    }
    if (read_res1.pattern.empty()) {
        MethylData res = meth_pattern_count(read_res2.pattern);
        res.pair_snps = read_res2.read_snps;
        return res;
    }
    if (read_res2.pattern.empty()) {
        MethylData res = meth_pattern_count(read_res1.pattern);
        res.pair_snps = read_res1.read_snps;
        return res;
    }
    ReadMethData read_res = merge(read_res1, read_res2);
    if (read_res.read_first_ind == -1){
        MethylData empty_res;
        empty_res.countMethyl = 0;
        empty_res.countUnmethyl = 0;
        empty_res.pattern = "";
        return empty_res;
    }
    MethylData res = meth_pattern_count(read_res.pattern);
    int index1 = 0;
    int index2 = 0;
    std::vector<ReadSNP> res_snps;
    ReadSNP snp1;
    ReadSNP snp2;
    while (index1 < read_res1.read_snps.size() && index2 < read_res2.read_snps.size()){
        snp1 = read_res1.read_snps[index1];
        snp2 = read_res2.read_snps[index2];
        if (snp2.locus > snp1.locus){
            res_snps.push_back(snp1);
            index1++;
//            snp1 = read_res1.read_snps[index1];
        }
        if (snp1.locus > snp2.locus){
            res_snps.push_back(snp2);
            index2++;
//            snp2 = read_res2.read_snps[index2];
        }
        if (snp2.locus == snp1.locus){
            if(snp2.snp_let == snp1.snp_let){
                res_snps.push_back(snp2);
            } else {
                ReadSNP empty_snp;
                empty_snp.locus = snp2.locus;
//                res_snps.push_back(empty_snp);
            }
            index1++;
            index2++;
        }
    }
    while (index1 < read_res1.read_snps.size()){
        snp1 = read_res1.read_snps[index1];
        res_snps.push_back(snp1);
        index1++;
    }
    while (index2 < read_res2.read_snps.size()){
        snp2 = read_res2.read_snps[index2];
        res_snps.push_back(snp2);
        index2++;
    }
    res.pair_snps = res_snps;
//2 is the index of the meth_pattern
    return res;
}

MethylData SnpMethHist::meth_pattern_count(std::string meth_pattern) {
    MethylData res;
    int countMethyl = 0;
    int countUnmethyl = 0;
    for (int i = 0; i < meth_pattern.length(); i++){
        char cur_char = meth_pattern[i];
        if(cur_char == METH){
            countMethyl++;
        } else if (cur_char == UNMETH){
            countUnmethyl++;
        }
    }
    res.countMethyl = countMethyl;
    res.countUnmethyl = countUnmethyl;
    res.pattern = meth_pattern;
    return res;
}


ReadMethData SnpMethHist::samLineToPatVec(std::vector <std::string> tokens) {
/** Given tokens of a sam line:
*   QNAME, FLAG, RNAME (chrom), POS, MAPQ, CIGAR, RNEXT,
*   PNEXT, TLEN, SEQ, QUAL, and possibly more.
*  return a new vector with the following fields:
*  [chr, first_CpG_ind, meth_pattern, start_loc, length]
*
*  In case line is empty or invalid, return an empty vector,
*  and update the corresponding counter
*  */
    ReadMethData empty_res;
    if (tokens.empty()) {
        return empty_res;
    }
    try {

        if (tokens.size() < 11) {
            throw std::invalid_argument("too few arguments in line");
        }
        unsigned long start_locus = stoul(tokens[3]);   // Fourth field from bam file
        std::string cur_chrom = tokens[2];
        int samflag = stoi(tokens[1]);
        std::string seq = tokens[9];
//        std::string qual_str = tokens[10];
//std::string bp_qual = tokens[10];
        std::string CIGAR = tokens[5];

        //TODO add quality filter
//        int qual = int(tokens[snp_index]) - 33;
        int qual = stoi(tokens[4]);
        if (qual < qual_filter){
            return empty_res;
        }

        seq = clean_CIGAR(seq, CIGAR);
        unsigned long int cur_end = std::max((long int)1, (long int) fasta_end - (long int) MAX_READ_LEN);
        if (start_locus > cur_end){
            load_fasta(cur_chrom, start_locus);
        }

        std::string ref = genome_ref.substr(start_locus - offset - 1, seq.length());

        std::string meth_pattern;

        ReadMethData read_res = compareSeqToRef(seq, ref, start_locus, samflag, meth_pattern);
        int start_site = read_res.read_first_ind;

        if (start_site < 1) {
            readsStats.nr_empty++;
            return read_res; // return empty vector
        }

        read_res.chrom = tokens[2];

        return read_res;
    }
    catch (std::exception &e) {
        std::string msg = "[ " + chr + " ] " + "Exception while processing line "
                          + std::to_string(line_i) + ". Line content: \n";
        std::cerr << "[ SnpMethHist ] " << msg;
        print_vec(tokens);
        std::cerr << "[ SnpMethHist ] " << e.what() << std::endl;
        readsStats.nr_invalid++;
    }
    return empty_res; // return empty vector
}

bool are_paired(std::vector <std::string> tokens1,
                std::vector <std::string> tokens2) {
// return true iff the reads are non empty and paired
    return ((!(tokens2.empty())) &&
            (!(tokens1.empty())) &&
            (tokens1[0] == tokens2[0]));
}

void SnpMethHist::proc2lines(std::vector <std::string> &tokens1,
                              std::vector <std::string> &tokens2) {

    try {
// sanity check: two lines must have the same QNAME
        if ((!(tokens2.empty())) &&
            (!(tokens1.empty())) &&
            (tokens1[0] != tokens2[0])){
            readsStats.nr_invalid += 2;
            throw std::invalid_argument("lines are not complements!");
        }

// Merge 2 complementary lines to a single output.
        MethylData res = merge_and_count_methyl_data(samLineToPatVec(tokens1), samLineToPatVec(tokens2));
        if (res.countMethyl + res.countUnmethyl >= min_cpg) {
            for (ReadSNP rs : res.pair_snps){
                SnpObj cur_snp = locus2SNP(rs.locus);
                int start_index;
                if (rs.snp_let == cur_snp.snp_let1){
                    start_index = 0;
                } else if (rs.snp_let == cur_snp.snp_let2) {
                    start_index = 3;
                } else {
                    continue;
                }
                float cur_prop = (float) res.countMethyl / (float) (res.countMethyl + res.countUnmethyl);
                if (cur_prop < range[0])
                    continue;
                int bin_ind = 0;
                for (bin_ind = 0; bin_ind < range.size() - 1; bin_ind++) {
                    if ((cur_prop >= range[bin_ind]) && (cur_prop < range[bin_ind + 1])) {
                        break;
                    }
                }
                if (bin_ind == range.size() - 1) {
                    bin_ind--;
                }
                snp_hist[rs.list_index][start_index + bin_ind] += 1;
            }
            readsStats.nr_short++;
            return;
        }

    }
    catch (std::exception &e) {
        std::string msg = "[ " + chr + " ] Exception while merging. lines ";
        msg += std::to_string(line_i) + ". Line content: \n";
        std::cerr << "[ SnpMethHist ] " << msg;
        print_vec(tokens1);
        print_vec(tokens2);
        std::cerr <<  "[ SnpMethHist ] " << e.what() << std::endl;
        return;
    }
}

/***************************************************************
 * *
 * print stats *
 * *
 ***************************************************************/

void SnpMethHist::print_stats_msg() {
/** print informative summary message */

    int sucess = line_i ? int((1.0 - ((double) readsStats.nr_invalid / line_i)) * 100.0) : 0;

    std::string msg = "[ " + chr + " ] ";
    msg += "finished " + addCommas(line_i) + " lines. ";
    if (is_paired_end) {
        msg += "(" + addCommas(readsStats.nr_pairs) + " pairs). ";
    }
    msg += addCommas(line_i - readsStats.nr_empty - readsStats.nr_invalid) + " good, ";
    msg += addCommas(readsStats.nr_empty) + " empty, ";
    if (min_cpg > 1) {
        msg += addCommas(readsStats.nr_short) + " with too few CpGs. ";
    }
    msg += addCommas(readsStats.nr_invalid) + " invalid. ";
    msg += "(success " + std::to_string(sucess) + "%)\n";
    std::cerr << "[ SnpMethHist ] " << msg;
}

/***************************************************************
 * *
 * Init *
 * *
 ***************************************************************/

bool SnpMethHist::first_line(std::string &line) {
// find out if the input is paired- or single-end
    try {
// find out if single- or paired-end
        std::vector <std::string> tokens = line2tokens(line);
        auto flag = (uint16_t) stoi(tokens.at(1));
        chr = tokens.at(2);
        return (flag & 1); // file is paired end iff flag 0x1 is on
    }
    catch (std::exception &e) {
        std::cerr << "[ SnpMethHist ] " << "[ " + chr + " ]" << "Invalid first line: \n" << line;
        std::cerr << "\nexception: " << e.what() << std::endl;
        throw e;
    }
}

void SnpMethHist::initialize_patter(std::string &line_str) {
// first line based initializations
    if (!(dict.empty())) { return; }
    is_paired_end = first_line(line_str);
    load_genome_ref();
}

/***************************************************************
 * *
 * Parse bam   *
 * *
 ***************************************************************/

void SnpMethHist::print_progress(){
    if (line_i && !(line_i % 100000)){
        clock_t tock = clock();
        double elapsed_secs = double(tock - tick) / (CLOCKS_PER_SEC * 60);
        tick = tock;
        std::cerr << "[ SnpMethHist ] [ " + chr + " ]" << " line " << addCommas(line_i)
                  << " in " << std::setprecision(2) << elapsed_secs << " minutes." << std::endl;
    }
}

void SnpMethHist::proc1line(std::vector <std::string> &tokens1) {
    proc2lines(tokens1, dummy_tokens);
    tokens1.clear();
}

void SnpMethHist::proc_sam_in_stream(std::istream& in){
    std::vector <std::string> tokens1, tokens2;
    for (std::string line_str; std::getline(in, line_str); line_i++){
        print_progress();

        if (line_str.empty()) { continue; }

        initialize_patter(line_str);

        if (tokens1.empty()) {
            tokens1 = line2tokens(line_str);
            if (!is_paired_end) {
                proc1line(tokens1);
                tokens1.clear();
            }
            continue;
        }
        tokens2 = line2tokens(line_str);
        if (are_paired(tokens1, tokens2)) {
            proc2lines(tokens1, tokens2);
            readsStats.nr_pairs++;
            tokens1.clear();
        } else {
            proc1line(tokens1);
            tokens1 = tokens2;
        }
    }
    if (! tokens1.empty()) { proc1line(tokens1); }
    print_stats_msg();
    dump_snphist();
}

void SnpMethHist::action(std::string filename) {
    std::ifstream samFile(filename, std::ios::in);

    if (!(samFile)){
        proc_sam_in_stream(std::cin);
    } else if (samFile.is_open()) {
        proc_sam_in_stream(samFile);
        samFile.close();
    }
}


/***************************************************************
 * *
 * Main *
 * *
 ***************************************************************/

/*
 * Input Arguments Parsering class
 */
class InputParser{
public:
    InputParser (int &argc, char **argv){
        for (int i=1; i < argc; ++i)
            this->tokens.emplace_back(std::string(argv[i]));
    }
    const std::string& getCmdOption(const std::string &option) const{
        std::vector<std::string>::const_iterator itr;
        itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()){
            return *itr;
        }
        static const std::string empty_string;
        return empty_string;
    }
    bool cmdOptionExists(const std::string &option) const{
        return std::find(this->tokens.begin(), this->tokens.end(), option)
               != this->tokens.end();
    }
private:
    std::vector <std::string> tokens;
};

bool is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

std::string get_param_str(InputParser &input, std::string name, std::string defval) {
    std::string param_str = input.getCmdOption(name);
    if (!param_str.empty()) {
        return param_str;
    }
    return defval;
}

std::vector<float> parse_range(std::string &range_str) {
    std::vector<float> vect;

    std::stringstream ss(range_str);

    for (float i; ss >> i;) {
        vect.push_back(i);
        if (ss.peek() == ',')
            ss.ignore();
    }
    float tmp = -1;
    for (std::size_t i = 0; i < vect.size(); i++) {
        if (vect[i] <= tmp) {
            std::cout << "Invalid range - non monotonic: " << range_str << std::endl;
            throw std::invalid_argument("Invalid range");
        }
        if ((vect[i] < 0) || (vect[i] > 1)) {
            std::cout << "Invalid range - must be in [0,1]: " << vect[i] << std::endl;
            throw std::invalid_argument("Invalid range");
        }
        tmp = vect[i];
//        std::cout << vect[i] << std::endl;
    }
    if ((vect[0] > 0) || (vect[vect.size() - 1] < 1)) {
        std::cout << "Invalid range - must start with 0 and end with 1."  << std::endl;
        throw std::invalid_argument("Invalid range");
    }
    return vect;
}


int main(int argc, char **argv) {
//    std::cout << "Yello 0."  << std::endl;
    try {
//        std::cout << "Yello 1."  << std::endl;
        InputParser input(argc, argv);
        int min_cpg = 1;
        int quality_filter = 0;
        if (input.cmdOptionExists("--min_cpg")){
            std::string min_cpg_string = input.getCmdOption("--min_cpg");
            if ( !is_number(min_cpg_string) )
                throw std::invalid_argument("Invalid min_cpg argument. Should be a non-negative integer.");
            min_cpg = std::stoi(min_cpg_string);
        }
        int clip = 0;
        int smart_clip = 0;
        if (input.cmdOptionExists("--clip")){
            std::string clip_str = input.getCmdOption("--clip");
            if ( !is_number(clip_str) )
                throw std::invalid_argument("Invalid clip argument. Should be a non-negative integer.");
            clip = std::stoi(clip_str);
        }
        if (input.cmdOptionExists("--smart_clip")){
            std::string clip_str = input.getCmdOption("--smart_clip");
            if ( !is_number(clip_str) )
                throw std::invalid_argument("Invalid smart_clip argument. Should be a non-negative integer.");
            smart_clip = std::stoi(clip_str);
        }
        if (input.cmdOptionExists("--qual_filter")){
            std::string qual_str = input.getCmdOption("--qual_filter");

            if ( !is_number(qual_str) )
                throw std::invalid_argument("invalid qual_filter argument. qual_filter should be a non-negative integer.");
            quality_filter = std::stoi(qual_str);
        }
        std::string range_str = get_param_str(input, "-r", "");
        if (!range_str.size()) {
            std::cout << "You must specify a range" << std::endl;
            return 1;
        }
        if (argc < 3) {
            throw std::invalid_argument("Usage: SnpMethHist CPG_REF FASTA_REF SNP_FILE --min_cpg -r (Ranges of methylation average, in [0,1]. For example: 0,0.2001,0.8,1.)");
        }
        std::vector<float> range = parse_range(range_str);
        SnpMethHist p(argv[1], argv[2], argv[3], argv[4], min_cpg, quality_filter, range, smart_clip);
        p.action("");

    }
    catch (std::exception &e) {
        std::cerr << "[ SnpMethHist ] Failed! exception:" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }
//std::cerr << elapsed_secs << std::endl;
    return 0;
}