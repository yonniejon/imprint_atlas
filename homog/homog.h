//
// Created by nloyfer on 5/27/19.
//

#ifndef PATS_READS_LENS_PAT2RLEN_H
#define PATS_READS_LENS_PAT2RLEN_H

//#include <boost/algorithm/string/predicate.hpp>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <math.h>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <string>
#include <stdlib.h>
#include <algorithm>

#define SEP "\t"
#define DEFAULT_BLOCKS "/cs/cbio/netanel/blocks/outputs/blocks.s216.bed.gz"
#define DEFAULT_LEN "5"

class Block{
public:
    int start;
    int end;
    int id;

    Block(int s, int e, int ind){
        start = s;
        end = e;
        id = ind;
    }
};

class Homog {
    int32_t *counts;


    bool debug;
    std::string blocks_path;
    std::string output_prefix;
    std::vector<Block> borders;
    std::vector<int> borders_counts;
    std::vector<std::string> coords;
    std::vector<float> range;
    std::vector<Block> cur_block_queue;
    int nr_blocks = 0;
    int cur_block_ind = 0;
    int min_cpgs = 0;

    int nr_bins;

    int read_blocks();

    void dump(int *data, int width, std::string out_path);
    void update_m2(int block_ind, std::string pat, int count);
    void update_m2_blocks(std::string pat, int count, int block_id);

    int proc_line(std::vector <std::string> tokens);

    void update_block(int ind, std::string pat, int count);
    void update_blocks(std::string pat, int32_t count, int read_start, int read_end);

    int32_t* init_array(int len);

    void debug_print(int ind, std::vector<std::string> tokens);

    int blocks_helper(std::istream &instream);

public:
    Homog(std::string rlen_path, std::string blocks_path, std::vector<float> range,
            int m_order, bool deb);

    ~Homog();

    void parse(std::string pat_path);
};

bool hasEnding (std::string const &fullString, std::string const &ending);
#endif //PATS_READS_LENS_PAT2RLEN_H
