# /usr/bin/python3
# main functions and dataclasses for
# generating DNA  sequences with variant
# coordinates  and reference genome. Also
# include useful utility function for
# preprocessing the input files.
# Date: 15/01/2022
# Author: Zhexin Liu


import gzip
import kipoiseq
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import joblib
from kipoiseq import Interval
import pyfaidx
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
#----------- for prediction of expression -----------------#



def get_cons(*items, method='inter'):
    """return the intersection or union of multiple
    lists. method =inter / union
    output type: set"""

    # make them all the set types
    set_items = []
    for i, genes in enumerate(items):
        if len(genes) == len(set(genes)):
            set_items.append(set(genes))
        else:
            return ('WARNING: the {}th gene set contains duplicate genes.'.format(i + 1))

    num = len(items)
    cur_list = []
    cur_index = None
    for i, genes in enumerate(set_items):
        if cur_index == None:
            cur_list = genes;
            cur_index = i
            continue
        else:
            if method == 'inter': cur_list = cur_list.intersection(genes)
            if method == 'union': cur_list = cur_list.union(genes)
            cur_index = i
    return (cur_list)




def gene_extractor(gff_file, mapping_list, gzziped=False):
    """input: gff file, list of wanted gene names
       output: dict
    """

    def _open(file):
        return gzip.open(file, 'rt') if gzziped else open(file, 'r')

    def get_nonoverlap(chr_, start, end, gene_pos):
        """ return the end position of the last up stream
        non-overlapping gene"""
        return (max(np.array(gene_pos[chr_])[:, 1][np.array(gene_pos[chr_])[:, 1] < start]))

    with _open(gff_file) as f:
        up_gene = False; gene_pos = {}  # {(start, end), ...}
        last_gene = None; cur_gene = None
        gene_dict = {}
        for line in f:
            if line.startswith('#'): continue
            chr_, xx, seq_type, start, end, score, strand, frame, attri = line.strip().split('\t')
            if attri.split(';')[0].startswith('ID=gene-'):
                cur_gene = (attri.split(';')[0].split('ID=gene-')[1], chr_, start, end)

                if cur_gene[0] in mapping_list and last_gene != None and chr_ == last_gene[-3]:
                    # store upstream and current gene
                    gene_dict['up_border'] = get_nonoverlap(chr_, int(start),
                                                            int(end), gene_pos) +1
                    gene_dict['name'] = cur_gene[0]
                    gene_dict['seq_type'] = seq_type
                    gene_dict['chr'] = chr_
                    gene_dict['start'] = start
                    gene_dict['end'] = end
                    gene_dict['strand'] = strand
                    gene_dict['attri'] = attri
                    up_gene = True
                elif up_gene and gene_dict['chr'] == chr_ and int(start) > int(gene_dict['end']):
                    # store downstread gene (only the start of downstrad gene
                    #                       is bigger than the end of target
                    #                       gene)

                    gene_dict['down_border'] = int(start)-1
                    up_gene = False
                    gene_dict['chr'] = transform_chrom(gene_dict['chr'])
                    yield gene_dict
                    gene_dict = {}

                gene_pos.setdefault(chr_, []).append((int(start), int(end)))
                last_gene = cur_gene


def extract_TSS(gene_id, TSS_file):
    "TSS_fille: path of the TSS file."
    def _open(file):
        return pd.read_csv(TSS_file)
    f = _open(TSS_file) #  pd.dataframe
    TSS_array = f[f.loc[:,'Gene name' ] == gene_id].loc[:,'Transcription start site (TSS)']
    TSS_array = [i -1 for i in TSS_array]
    del f
    return TSS_array # list

def get_max_min(value_list, MAX = False, MIN = False):
    "return the index of max/min value in the list"
    value_list = list(value_list)
    if MAX:
        return value_list.index(max(value_list))
    elif MIN:
        return value_list.index(min(value_list))

def variants_extractor(vcf_path, Interval, acc_id, gzipped=False):
    """
    vcf: path of vcf;
    interval: kipoiseq.Interval object;
    """
    def _open(file):
        return gzip.open(file, 'rt') if gzipped else open(file, 'r')


    f= '{vcf_path}/{acc_id}.{chrom}.output.g.filtered.vcf{extention}'.format(vcf_path =  vcf_path,
                                                                  acc_id = acc_id,
                                                                  chrom = Interval.chrom,
                                                                  extention = '.gz' if gzipped else '')
    #since regions [0-98303] and [294912-393215] wouldn't be used, we only extract
    #variants that fall within 98304-294911.
    SEQUENCE_LENGTH = 393216
    START =  Interval.start + int(SEQUENCE_LENGTH/4)
    END = Interval.end - int(SEQUENCE_LENGTH/4)

    def get_less_noisy(alt_list, reads_list):
        af = np.array(reads_list)/sum(reads_list)
        index = get_max_min(np.abs(af[1:] - 0.5), MIN=True)
        return alt_list[index]

    with _open(f) as file:
        for line in file:
            if line.startswith('#'): continue
            chr_, pos, id, ref, alt_list, *_, gytpe, reads = line.strip().split('\t')
            # NOTE: start in Interval object is 0-based;
            #       start in VCF file is 1-based

            if  START  <= int(pos) -1 <= END:
                # some location might have two allels
                # take the one with allele frequency close to 0.5 or 1
                alt_list = alt_list.strip().split(',')
                reads = [int(i) for i in reads.split(':')[1].split(',')]


                # Replace '*' with '' so the sequence doesn't contain '*'
                alt_list = ['' if i == '*' else i for i in alt_list]
                #want_alt = alt_list[get_max_min(reads[1:], MAX=True)]
                want_alt = get_less_noisy(alt_list, reads)
                yield kipoiseq.dataclasses.Variant(chrom = chr_, pos= int(pos),
                                             ref=ref, alt=want_alt,
                                             id=id) # pos: 1-based


def transform_chrom(refseq_name):
    """transform the refseq assembly name to molecular name
    on reference GRCh37 """
    p = re.compile('[1-9]')
    index = p.search(refseq_name.strip().split('.')[0]).start()
    if refseq_name.strip().split('.')[0][index:] == '23':
        return 'X'
    elif refseq_name.strip().split('.')[0][index:] == '24':
        return 'Y'
    else:
        return refseq_name.strip().split('.')[0][index:]



class Gene:
    def __init__(self,
                 name:str,
                 seq_type:str,
                 chr:str,
                 start: str, # 1-based
                 end: str, # 1-based
                 strand: str,
                 attri:str,
                 down_border:int, # 1-based
                 up_border: int): # 1-based

        self.name = name
        self.seq_type = seq_type
        self.chrom = chr
        self.start = (int(start) - 1)
        self.end = (int(end) - 1)
        self.strand = strand
        self.info = attri
        self.down_border = down_border -1
        self.up_border = up_border - 1

    def width(self):
        return self.end - self.start + 1



def calculate_TSS(ref_pre, alt_pre):
    """input type: np.array, 896*5313
        remap the relative location of TSS to the output sequence,
        accroding to how many bps are chopped due to indels.
    """
    # find indels that are winthin the region
    index_range = [ ref_pre.shape[0]//2 - 2, ref_pre.shape[0]//2 -1, ref_pre.shape[0]//2 ]
    return np.sum((alt_pre - ref_pre)[index_range], axis=0)

def write_file(df, output_name):
    df.to_csv(output_name)