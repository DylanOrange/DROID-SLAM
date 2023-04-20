# Instruction for Students
It has to be noted that sharing my access of slurm with you is only for urgent usage. Please follow this instruction exactly step by step.
Basically, I'll create a shared terminal on my desktop. With this shared terminal, you have all the access that I can have. 

### Keep the shared link 100% confidential! Do not tell anyone, it is a secret!
### Do not mess up Xingxing's documents, and keep the security of all his documents! Do not read, write, modify anything except your own folder.
### Do not put your files anywhere except the created folders following this instruction!

### Instructions

- Give me your public ssh keys, and I'll create a shared terminal via Tmate for your access.

- Create your personal folder named "tmp_zuox_yourname" at "/usr/wiss/zuox/storage/slurm/zuox/", and git pull your code here.

- If you need to create Conda environment, please specify the name of your environment like "tmp_yourname_xxxxx"

- Here are some instructions for sharing your training data with me. If the dataset is not large-size (below 100G), you can download it at 
"~/storage/user/zuox/tmp_yourname_data/". If the dataset is really big, do not download it again to save storage space. You can share it with me by the following commands:
```bash
# Example: student lud intends to share the folder "/storage/user/lud/lud/DeFlowSLAM/datasets" to me
# Student lud needs to do this on his own terminal: 
# (Actually, it is much faster to 'chmod g+x,o+x directory_by_directory from top to bottom' )
$ chmod -R g+x,o+x /storage/user/lud/lud/DeFlowSLAM  
# This might take some time, please be patient
$ setfacl -R -m user:zuox:rwx /storage/user/lud/lud/DeFlowSLAM/datasets/

# On my shared Tmate terminal, I (Xingxing) have the access to the shared folder by
$ cd /storage/user/lud/lud/DeFlowSLAM/datasets
# Create a soft link at your temp folder "/usr/wiss/zuox/storage/slurm/zuox/tmp_zuox_yourname"
$ ln -s  /storage/user/lud/lud/DeFlowSLAM/datasets  /usr/wiss/zuox/storage/slurm/tmp_zuox_yourname/datasets
```

- [!] Reminder of the storage on Slurm:
  1. The storage space of your home folder is asked to be under 100 G, but you can save the large-size dataset in ~/storage/user/yourname (HDD)
  2. The storage space with fast access speed is ~/storage/slurm/zuox/tmp_zuox_yourname (SSD), do not put large-size dataset here. However, it is **highly recommended** to save you slurm log file (specified in .sbatch file by "SBATCH --output=" and "SBATCH --error="), and save the on-the-fly checkpoints there. It benifit the training speed.
  
- Notice about the slurm job email notifications
  I do not wanna to be disturbed by your submitted jobs. You can disable the email notifications. Or in the .sbatch file, please inform yourself instead of me by
```bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YourChairID
```
- Check your GPU memory usage in Python
```python
def fmt(i):
    reserved_gpu = torch.cuda.memory_reserved(i)
    total_gpu = torch.cuda.get_device_properties(i).total_memory
    return f'{100 * reserved_gpu / total_gpu:.1f}'


gpu_usage = [fmt(i) for i in range(torch.cuda.device_count())]
                gpu_usage = ",".join(gpu_usage)
print(gpu_usage)

```