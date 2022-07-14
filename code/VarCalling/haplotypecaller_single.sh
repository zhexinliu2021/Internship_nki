#  prototype for calling mutations from WGS from CCLE. 
#  Computing envirnment: Lisa, SURF. (lisa.surfsara.nl)
#  WGS: http://www.ncbi.nlm.nih.gov/sra?term=PRJNA523380     
#------------------------------------------------------------
#!/bin/bash
#Set job requirements
#SBATCH -p gpu_shared
#SBATCH --gpus=1
#SBATCH --job-name=hc_single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH -t 60:00:00
#SBATCH --output=slurm_hC_single.new_%A.out

SECONDS=0
module load 2021
module load SAMtools/1.12-GCC-10.3.0 

#Copy input files to scrath system.
mkdir "$TMPDIR"/input_dir 
cp  $HOME/bam_files/one_sample/Homo_sapiens_assembly19*   "$TMPDIR"/input_dir
cp /project/lzhexin/output_dir_m/*  "$TMPDIR"/input_dir
echo "Copying input files finished!"


#create output file on scratch.
mkdir "$TMPDIR"/output_dir_haplotypecaller_single
echo "create output dir finished"

#run haplotypecaller 
#gatk --java-options "-Xmx4g" HaplotypeCaller  \
#   -R Homo_sapiens_assembly38.fasta \
#   -I input.bam \
#   -O output.g.vcf.gz \
#   -ERC GVCF 
cd "$TMPDIR"/input_dir

gatk --java-options "-Xmx30g" HaplotypeCaller  \
   -R Homo_sapiens_assembly19.fasta \
   -I SRR8788980.1.new.bam \
   -O "$TMPDIR"/output_dir_haplotypecaller_single/SRR8788980.1.new.output.g.vcf.gz \
   -native-pair-hmm-threads 6
#   -ERC GVCF

echo "haplotypecaller finished"

#Copy output dir back to home.
cp -r "$TMPDIR"/output_dir_haplotypecaller_single  /project/lzhexin/

echo "copying output dir finished"


duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."





