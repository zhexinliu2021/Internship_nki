import numpy as np
import pandas as pd
import gzip
import kipoiseq
import re
import tensorflow as tf
import tensorflow_hub as hub
import joblib
from kipoiseq import Interval
import pyfaidx
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns



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


    f= '{vcf_path}/{acc_id}.1.new.{chrom}.output.g.vcf{extention}'.format(vcf_path =  vcf_path,
                                                                  acc_id = acc_id,
                                                                  chrom = Interval.chrom,
                                                                  extention = '.gz' if gzipped else '')
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

            if  Interval.start  <= int(pos) -1 <= Interval.end:
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
        self.start = int(start) - 1
        self.end = int(end) - 1
        self.strand = strand
        self.info = attri
        self.down_border = down_border -1
        self.up_border = up_border - 1

    def width(self):
        return self.end - self.start + 1






class Enformer:

    def __init__(self, tfhub_url):
        self._model = hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence,
                                target_mask, output_head='human'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            prediction = tf.reduce_sum(
                target_mask[tf.newaxis] *
                self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)


class EnformerScoreVariantsRaw:

    def __init__(self, tfhub_url, organism='human'):
        self._model = Enformer(tfhub_url)
        self._organism = organism

    def predict_on_batch(self, inputs):
        ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
        alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]

        return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


class EnformerScoreVariantsNormalized:

    def __init__(self, tfhub_url, transform_pkl_path,
                 organism='human'):
        assert organism == 'human', 'Transforms only compatible with organism=human'
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            transform_pipeline = joblib.load(f)
        self._transform = transform_pipeline.steps[0][1]  # StandardScaler.

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)


class EnformerScoreVariantsPCANormalized:

    def __init__(self, tfhub_url, transform_pkl_path,
                 organism='human', num_top_features=500):
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            self._transform = joblib.load(f)
        self._num_top_features = num_top_features

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)[:, :self._num_top_features]

# TODO(avsec): Add feature description: Either PCX, or full names.

# @title `variant_centered_sequences`

class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def variant_generator(vcf_file, gzipped=False):
    """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""

    def _open(file):
        return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)

    with _open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            chrom, pos, id, ref, alt_list = line.split('\t')[:5]
            # Split ALT alleles and return individual variants as output.
            for alt in alt_list.split(','):
                yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                                   ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def variant_centered_sequences(vcf_file, sequence_length, gzipped=False,
                               chr_prefix=''):
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
        reference_sequence=FastaStringExtractor(fasta_file))

    for variant in variant_generator(vcf_file, gzipped=gzipped):
        interval = Interval(chr_prefix + variant.chrom,
                            variant.pos, variant.pos)
        interval = interval.resize(sequence_length)
        center = interval.center() - interval.start

        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, [variant], anchor=center)

        yield {'inputs': {'ref': one_hot_encode(reference),
                          'alt': one_hot_encode(alternate)},
               'metadata': {'chrom': chr_prefix + variant.chrom,
                            'pos': variant.pos,
                            'id': variant.id,
                            'ref': variant.ref,
                            'alt': variant.alt}}



def plot_tracks(tracks, interval, height=1.5):
  fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
  for ax, (title, y) in zip(axes, tracks.items()):
    ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
  ax.set_xlabel(str(interval))
  plt.tight_layout()


def plot_full_track(tracks, interval, up_border, down_border, height= 1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        arry = np.insert(np.linspace(interval.start, interval.end, num=len(y)), 0, int(up_border))
        arry = np.insert(arry, len(arry), int(down_border))
        arry_y = np.insert(y, 0, 0)
        arry_y = np.insert(arry_y, len(arry_y), 0)

        ax.fill_between(arry, arry_y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(f'{interval.chrom}:{up_border}-{down_border}')
    plt.tight_layout()

def calculate_TSS(ref_pre, alt_pre):
    "input tuype: np.array, 983*5313"
    index_range = [ ref_pre.shape[0]//2 - 1, ref_pre.shape[0]//2, ref_pre.shape[0]//2 +1]
    return np.sum((alt_pre - ref_pre)[index_range], axis=0)



