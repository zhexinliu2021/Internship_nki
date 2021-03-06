# THis software can let you take input a VCF file
# and filter out allels that have less alt reads
# or total reads covery. Usage as below
# Date: 11/02/2022

# usage: Filter_Rs.py [-h] [-i INPUT] [-o OUTPUT] [-r] [-S] [-IN]
# optional arguments:
#   -h, --help            show this help message and exit
#   -i INPUT, --input INPUT
#                         Input vcf file's dir
#   -o OUTPUT, --output OUTPUT
#                         output vcf file's dir (gzip file)
#   -r, --reads_file      file contains A_reads/total_reads
#   -S, --SNP             flag indicating whether to call snps
#   -IN, --INDELS         flag indicating whether to call indels

# /usr/bin/python3
import argparse
import pandas as pd
import numpy as np
import gzip
#from tqdm import tqdm



def main():

    def _open(file):
        return gzip.open(file,'rt') if file.endswith('.gz') else open(file, 'r')


    vcf_i = _open(in_dir)
    vcf_o = gzip.open(out_dir, 'wb')
    fields = ''
    chr_list = []; ratio_list = []

    for line_index, line in enumerate(vcf_i):

        filterd = 'yes'
        if line.startswith("#CHROM"):

            fields = line[1:].split("\t"); filterd = 'no' # dont filter headers
            #print(fields)
            #print(line_index)
        elif line.startswith('##'): filterd = 'no'; #print(line_index)
        elif not line.startswith('#CHROM') and not line.startswith('##') :
            format = fields[-2]; Gt = fields[-1]
            mapping = dict(list(zip(fields,line.split())))
            gt_map = dict(list(zip(mapping[format].split(":"), mapping[Gt].split(":"))))
            # CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	DAN-G
            #GT: AD:DP: GQ:PL      0 / 1: 6, 4: 10:99: 105, 0, 218
            # step.1: if the number of reads is enough
            if int(gt_map['DP']) < min_DP: continue

            #step 2: check if the allel's reads are enough.
            AD_mapping = dict(list(zip(mapping['ALT'].split(','), gt_map['AD'].split(',')[1:])))
            # (ALT,
            alt_list = mapping['ALT'].split(',')
            for alle in alt_list:
                AD = AD_mapping[alle]
                # if it is a SNP
                # with at least 4 reads coverage and two reads supporting allel, and allel frequency above 0.1
                if len(alle) == len(mapping['REF']) and mapping['REF'] != '*' \
                        and int(AD) >= min_AD and (int(AD)/int(gt_map['DP'])) >= min_ratio:
                    filterd = 'no'
                    if r_file :  #if reads model turned on, append the chr_list and ratio list.
                        chr_list.append(mapping['CHROM'])
                        ratio_list.append( int(AD)/int(gt_map['DP']))


                elif (len(alle) != len(mapping['REF']) or mapping['REF'] == '*') and int(AD) >= min_AD and indel_call: filterd = 'no'
                # if it is a indel and has reads >= min_AD, and model turned on, then don't filter the allel
            #print(line_index)
        #write to output file
        if filterd == 'no': vcf_o.write(line.encode("ascii"))
        #f.write(content.encode("ascii"))
    vcf_o.close()
    if r_file :
        df = pd.DataFrame({'CHR':np.array(chr_list), 'Aellel_ratio' : np.array(ratio_list) })
        df.to_csv("reads_ratio_DP.{}_AD.{}_af.{}.csv".format(min_DP, min_AD, min_ratio))


    vcf_i.close()











if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help= "Input vcf file's dir")
    parser.add_argument("-o","--output", type = str,
                        help= "output vcf file's dir (gzip file)")
    parser.add_argument('-r', "--reads_file", action="store_true", default=False,
                        help='file contains A_reads/total_reads')

    parser.add_argument('-S', "--SNP", action = "store_true", default= True,
                        help= "flag indicating whether to call snps")
    parser.add_argument('-IN', '--INDELS', action='store_true', default= False,
                        help= "flag indicating whether to call indels")




    args = parser.parse_args()

    in_dir = args.input; out_dir = args.output
    snp_call=args.SNP; indel_call = args.INDELS
    r_file = args.reads_file

    min_DP = 4
    min_AD = 2
    min_ratio = 0.0

    main()
