#!/bin/bash
#-------------------------
# Shell code to create the script that runs the mutation 
# calling in parallel on Lisa, Surf. 
# 

## configuration
TIME='05:00:00'



#step.1 how many tasks to assign
NPROC=6 # number of cores
CNTSK=1

Job_file="$PWD"

IFS=$'\n'
CHR_LIST=($(sed -n '/^@SQ/p'  /project/lzhexin/output_dir_m/split_bams/header | awk '{print $2}' | awk -F":" '{print $2}'))

NTSK=$(sed -n '/^@SQ/p'  /project/lzhexin/output_dir_m/split_bams/header | awk '{print $2}' | awk -F":" '{print $2}' | wc -l) #total numbe of tasks(chrs) 

#create job_file where the scripts are created.
mkdir "$PWD"/job_file/

while [ $CNTSK -lt $[$NTSK+$NPROC] ]
do
    if  [ $[$CNTSK+$NPROC-1] -le $NTSK ] 
    then    
        # one job script with 6 tasks (cores)
        #make one job script for each node
        # [1 - 1+NPROC], [1+NPROC - 1+2*NPROC], ...
        END=$[$CNTSK+$NPROC-1]

    else
        END=$NTSK
    fi

    CFILE="$PWD"/job_file/hC_GVCF."$CNTSK"_${END}.sh
    INDIR="input_dir_""$CNTSK"\_${END}
    OUTDIR="output_dir_${CNTSK}_${END}"
    TRANGE="${CNTSK}_${END}"
    start_i=$[$CNTSK-1]
    CHR_arry=("${CHR_LIST[@]:$start_i:$[$END-$CNTSK+1]}")
    touch $CFILE
    
    # dependencies {{{
    echo "#!/bin/bash">>$CFILE
    echo "#Set job requirements">>$CFILE
    echo "#SBATCH -p gpu_shared">>$CFILE
    echo "#SBATCH --gpus=1">>$CFILE
    echo "#SBATCH --job-name=hC_${TRANGE}" >>$CFILE
    echo "#SBATCH --ntasks=1">>$CFILE
    echo "#SBATCH --cpus-per-task="$NPROC >>$CFILE
    echo "#SBATCH -t "$TIME>>$CFILE
    echo "#SBATCH --output=slurm_hC.GVCF.${TRANGE}_%A.out">>$CFILE
    echo "#SBATCH --mem=100G">>$CFILE
    echo "SECONDS=0" >>$CFILE
    echo "module load 2021">>$CFILE
    echo "module load SAMtools/1.12-GCC-10.3.0">>$CFILE
    echo $'\n\n'>>$CFILE
    #}}}
    
    #create input dir
    echo "#create input dir on scratch sys">>$CFILE
    echo "mkdir \"\$TMPDIR\"/input_dir_${TRANGE}">>$CFILE

    # Copying bam and reference  genomes to scratch sys {{{
    echo "# copy bam and refernece genoemes to scratch sys">>$CFILE

    #create bam file
    echo "cp /project/lzhexin/output_dir_m/SRR8788980.1.new.bam*  \"\$TMPDIR\"/"$INDIR >>$CFILE

    for chr in "${CHR_arry[@]}";do
        #copy reference genomes and indexes
        #echo "${chr}"
        echo "cp  \${HOME}/bam_files/reference/references/Homo_sapiens_assembly19.${chr}.fasta  \${HOME}/bam_files/reference/references/index_dir/Homo_sapiens_assembly19.${chr}.*  \"\$TMPDIR\"/${INDIR}" >>$CFILE
    
    done

    echo "#creation of bam files and genome files done.">>$CFILE
    echo $'\n\n'>>$CFILE
    # }}}
    
    # create output dir {{{ 
    echo "#create output dir">>$CFILE
    echo "mkdir \"\$TMPDIR\"/${OUTDIR}">>$CFILE
    echo $'\n\n'>>$CFILE

    # }}}
   
    #create log files for each task {{{
    echo "#create log file dir">>$CFILE
    echo "mkdir \"\$SLURM_SUBMIT_DIR\"/logs_${TRANGE}" >>$CFILE
    echo $'\n\n'>>$CFILE

    # }}}
   
    #Mut calling {{{
    #split bam-> make bam.bai -> run hC
    echo "cd \"\$TMPDIR\"/${INDIR}">>$CFILE
    echo $'\n'>>$CFILE

    
    for chr in "${CHR_arry[@]}"; do
        echo "#--------- THIS IS chr ${chr} -----------">>$CFILE
        #split bam
        echo "( samtools view -b -@ 3 SRR8788980.1.new.bam  ${chr} > \"\$TMPDIR\"/${INDIR}/SRR8788980.1.new.${chr}.bam">>$CFILE
        echo "echo \"splitting bam file finished\"">>$CFILE

        #indexing bam
        echo "samtools index -@ 3 SRR8788980.1.new.${chr}.bam  \"\$TMPDIR\"/${INDIR}/SRR8788980.1.new.${chr}.bam.bai">>$CFILE
        echo "echo \"indexing bam finished\"">>$CFILE

        #gatk
        echo "gatk --java-options \"-Xmx60g\" HaplotypeCaller -R Homo_sapiens_assembly19.${chr}.fasta  -I  SRR8788980.1.new.${chr}.bam  -L ${chr}  -O  \"\$TMPDIR\"/${OUTDIR}/SRR8788980.1.new.${chr}.output.g.vcf.gz  -native-pair-hmm-threads 4">>$CFILE
        echo " echo \"gatk\" done">>$CFILE
        echo ") > \"\$SLURM_SUBMIT_DIR\"/logs_${TRANGE}/chr${chr}.log 2>&1 &">>$CFILE
        echo $'\n'>>$CFILE
    done
# }}}
    echo "wait">>$CFILE 
    echo $'\n\n'>>$CFILE

    #cp output {{{
    #copying output vcf to project space
    echo "cp -r \"$TMPDIR\"/${OUTDIR}  /project/lzhexin/SRR8788980/hC_single ">>$CFILE
    # }}}
    echo "duration=\$SECONDS">>$CFILE
    echo "echo  \"\$((\$duration / 60)) minutes and \$((\$duration % 60)) seconds elapsed.\" ">>$CFILE

    CNTSK=$[ $CNTSK + $NPROC ]
    
    
done









