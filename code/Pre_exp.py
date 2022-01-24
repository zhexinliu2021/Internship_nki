# /usr/bin/python3
#usage: python3 Pre_exp.py -i acc_id -o output

import numpy as np
import pandas as pd
import gzip
import kipoiseq
import importlib
import Models
import ToolsForExPrediction as tools
import matplotlib
import tensorflow as tf
import argparse
import os
import sys

def main():
    # mutation files' paths
    dam_file =  f'{tmp}/input/mutation_files/CCLE_mutations_bool_damaging.csv'
    hot_file = f'{tmp}/input/mutation_files/CCLE_mutations_bool_hotspot.csv'
    non_file = f'{tmp}/input/mutation_files/CCLE_mutations_bool_nonconserving.csv'
    con_file = f'{tmp}/input/mutation_files/CCLE_mutations_bool_otherconserving.csv'
    Con_file = f'{tmp}/input/mutation_files/Census_allSat.csv'

    # gene anotation path
    gff_file = f'{tmp}/input/files/GRCh37_latest_genomic.gff.gz'
    vcf_path = f'{tmp}/input/vcf'
    vcf_path = f'{vcf_path}/{acc_id}'

    # model paths
    transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
    model_path = 'https://tfhub.dev/deepmind/enformer/1'
    fasta_file = f'{tmp}/input/files/Homo_sapiens_assembly19.fasta'

    # Download targets from Basenji2 dataset
    targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
    df_targets = pd.read_csv(targets_txt, sep='\t')
    TSS_file = f'{tmp}/input/files/mart_export.txt.gz'




    # get target cancer genes
    #with tf.device(f'GPU:{gpu_id}'):
    tf.print( f"operating on GPU {gpu_id}", output_stream=sys.stdout )
    #print(f"operating on GPU {gpu_id}")

    gene_lists =  [ pd.read_csv(path).columns[1:].map(lambda x:x.split(' ')[0])
        for path in [dam_file, hot_file, non_file] ]

    Con_genes = pd.read_csv(Con_file).loc[:,'Gene Symbol']

    final_genes = tools.get_cons(tools.get_cons(*gene_lists, method='union'), Con_genes, method= 'inter')

    # extract the positions of each cancer gene
    gene_pos = tools.gene_extractor(gff_file = gff_file, mapping_list=final_genes, gzziped=True)

    # import enformer
    model = Models.Enformer(model_path)
    fasta_extractor = Models.FastaStringExtractor(fasta_file)
    tf.print("preprocessing files done", output_stream=sys.stdout)
    #print("preprocessing files done")

    # generate prediction for all genes
    SEQUENCE_LENGTH = 393216
    #acc_id = 'SRR8788980'

    # Make sure the GPU is enabled
    #assert tf.config.list_physical_devices('GPU'), ' -> GPU'
    #print("GPU enabled")


    #with tf.device(f'GPU:{gpu_id}'):
    df= pd.DataFrame([], columns= df_targets.description)
    for index, gene_dict in enumerate(gene_pos):
        # create Gene object
        gene = tools.Gene(**gene_dict)
        # create TSS interval objects list
        TSS_list = [kipoiseq.Interval(gene.chrom, TSS, TSS).resize(SEQUENCE_LENGTH) \
                    for TSS in tools.extract_TSS(gene_id=gene.name, TSS_file=TSS_file)]

        # prepare sequence for each TSS
        seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence = fasta_extractor)




        expression = np.zeros(5313)
        for TSS_item in TSS_list:
            #print(f"processing {TSS_item}")
            reference = seq_extractor.extract(TSS_item, [],
                                              anchor=TSS_item.center() - TSS_item.start)
            # generate variants list for each TSS
            variants = [i for i in tools.variants_extractor(vcf_path=vcf_path,
                                                             Interval=TSS_item, acc_id=acc_id, gzipped=True)]

            alternate = seq_extractor.extract(TSS_item, variants,
                                              anchor=TSS_item.center() - TSS_item.start)
            # make prediction on the 114,688-bp centering the TSS of the gene
            reference_prediction = model.predict_on_batch(Models.one_hot_encode(reference)[np.newaxis])['human'][0]
            alternate_prediction = model.predict_on_batch(Models.one_hot_encode(alternate)[np.newaxis])['human'][0]

            # 1* 5313
            expression = expression + tools.calculate_TSS(ref_pre=reference_prediction, alt_pre=alternate_prediction)

        expression = pd.DataFrame(expression.reshape(-1, 5313), columns=df_targets.description, index=[gene.name])
        df = pd.concat([df, expression], axis=0)
        tf.print(f"gene {gene.name}, NO. {index} done", output_stream=sys.stdout)
        #print(f"gene {gene.name}, NO. {index} done")
    tools.write_file(df, output_name= output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--acc_id", type=str,
                        help= "Input vcf file's dir")
    parser.add_argument("-o","--output", type = str,
                        help= "output vcf file's dir (gzip file)")
    parser.add_argument("-p", "--scratch", type=str,
                        help="temporary path")
    parser.add_argument("-gpu", "--gpu_id", type=str,
                        help="index of gpu (0-4)")
    args = parser.parse_args()
    acc_id = args.acc_id
    output_path = args.output
    gpu_id = args.gpu_id

    tmp=args.scratch
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    main()