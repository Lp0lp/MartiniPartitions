import sys
import MDAnalysis as md
import subprocess
import os
import numpy as np
from collections import Counter
from multiprocessing import Process
import alchemlyb
from alchemlyb.parsing.gmx import extract_dHdl
from alchemlyb.estimators import TI

partition_tools_base = "/local/lborge01/Martini_Partition_Source/"
m3_ffs               = '/local/lborge01/m3_ffs/'
states               = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def replace_words_in_file(original_file_path, new_file_path, words_to_replace, replacement_words):
    """
    Opens a file, replaces occurrences of multiple words with their respective replacements, 
    and writes the result to a new file.

    Parameters:
    original_file_path (str): The path to the original file.
    new_file_path (str): The path where the new file will be written.
    words_to_replace (list of str): The words to be replaced.
    replacement_words (list of str): The words to replace with.
    """
    if len(words_to_replace) != len(replacement_words):
        raise ValueError("The list of words to replace and the list of replacement words must be of the same length.")
    
    try:
        # Open the original file and read its contents
        with open(original_file_path, 'r') as file:
            file_contents = file.read()

        # Replace each word with its corresponding replacement
        for word_to_replace, replacement_word in zip(words_to_replace, replacement_words):
            file_contents = file_contents.replace(word_to_replace, replacement_word)

        # Write the modified contents to the new file
        with open(new_file_path, 'w') as file:
            file.write(file_contents)

        # print(f"Replaced {words_to_replace} with {replacement_words} and saved the result to '{new_file_path}'")
    except FileNotFoundError:
        print(f"The file {original_file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def staterunmd(resname, state, offset):
    try:
        os.makedirs(f'./{state}')
    except FileExistsError:
        pass

    ## Fix mdps and cp them 
    replace_words_in_file(f'{partition_tools_base}/mdps/template-CG_minimization-fep.mdp',
                          f'./{state}/min.mdp',
                         ['MOL','INIT-LAMBDA-STATE'],
                         [resname, str(state)])
    replace_words_in_file(f'{partition_tools_base}/mdps/template-CG_relaxation-fep.mdp',
                          f'./{state}/rel.mdp',
                         ['MOL','INIT-LAMBDA-STATE'],
                         [resname, str(state)])
    replace_words_in_file(f'{partition_tools_base}/mdps/template-CG_fep-vdw.mdp',
                          f'./{state}/fep.mdp',
                         ['MOL','INIT-LAMBDA-STATE'],
                         [resname, str(state)])

    ## Run the md
    os.chdir(f'./{state}')
    # offset=state + base_offset
    with open('./log.out', 'w') as output_file:
        subprocess.call("gmx grompp -f min.mdp -c ../../system.gro -p ../../system.top -o min.tpr -maxwarn 1", 
                        shell = True, stdout=output_file, stderr=subprocess.STDOUT)
        subprocess.call(f"gmx mdrun -deffnm min -v -nt 1 -pin on -pinoffset {offset} -pme cpu -pmefft cpu -bonded cpu -update cpu -nb cpu", 
                        shell = True, stdout=output_file, stderr=subprocess.STDOUT)
    
        subprocess.call("gmx grompp -f rel.mdp -c min.gro -p ../../system.top -o rel.tpr -maxwarn 2", 
                        shell = True, stdout=output_file, stderr=subprocess.STDOUT)
        subprocess.call(f"gmx mdrun -deffnm rel -v -nt 1 -pin on -pinoffset {offset} -pme cpu -pmefft cpu -bonded cpu -update cpu -nb cpu", 
                        shell = True, stdout=output_file, stderr=subprocess.STDOUT)
    
        subprocess.call("rm -f mdout.mdp min.* rel.cpt rel.edr rel.log rel.mdp rel.tpr rel.xtc rel.xvg", 
                        shell = True, stdout=output_file, stderr=subprocess.STDOUT)
    
        subprocess.call("gmx grompp -f fep.mdp -c rel.gro -p ../../system.top -o fep.tpr -maxwarn 2", 
                        shell = True, stdout=output_file, stderr=subprocess.STDOUT)
        subprocess.call(f"gmx mdrun -deffnm fep -v -nt 1 -pin on -pinoffset {offset} -pme cpu -pmefft cpu -bonded cpu -update cpu -nb cpu", 
                        shell = True, stdout=output_file, stderr=subprocess.STDOUT)
    
def check_processes(processes,offset,ncores=39):
    if len(processes)>=ncores:
        # print('Cleanup in place!')
        for proc in processes:
            proc.join()
        processes=[]
        offset = 0
        return processes, offset
    else:
        return processes, offset

def prepare_partition_setup(molname, cg_itp, cg_inputstructure, solvents, reps):
    
    resname              = np.unique(md.Universe(cg_inputstructure).atoms.resnames)[0] ## Molecule resname
    
    ###
    ###  Initialize the TI file structure
    ###
    for rep in reps:   ### Prep several reps
        workdir           = f"./Partitions/{molname}/{rep}"
        os.makedirs(workdir, exist_ok=True)
    
        
        ### Create system boxes & topology file
        ### 1 per solvent (Wat, Oct, Hexadec, Chloroform)
        for solvent in solvents:
            ## create solvent directory
            solvdir = f"{workdir}/{solvent}"
            os.makedirs(solvdir, exist_ok=True)
    
        
            ## Create sim box
            subprocess.call(f"gmx insert-molecules -f {partition_tools_base}/solvents/{solvent}.gro -ci {cg_inputstructure} -o {solvdir}/system.gro -nmol 1 -try 100000 -replace 'not resname {resname}' "
                            , shell = True)
        
            ## Write a topology file.
            header=[f'#include "{m3_ffs}/martini_v3.0.0.itp"\n',
                    f'#include "{m3_ffs}/martini_v3.0.0_solvents_v1.itp"\n',
                    f'#include "{m3_ffs}/martini_v3.0.0_ions_v1.itp"\n',
                    f'#include "{cg_itp}"\n',
                    '[system]\n',
                    f'TI: {solvent}\n',
                    '\n',
                    '[ molecules ]\n',
                    ]
        
            counts = Counter(md.Universe(f'{solvdir}/system.gro').atoms.residues.resnames)
            for item, count in counts.items():
                header.append(f"{item}    {count}\n")
        
            with open(f'{solvdir}/system.top', 'w+') as topout:
                for line in header:
                    topout.write(line)

def run_partitions_locally(molname, resname, reps, solvents):

    ## initialize cpu counter and process list    
    processes = []  
    offset = 0
    start_dir = os.getcwd()
    for rep in reps:
        workdir           = f"./Partitions/{molname}/{rep}"
        for solvent in solvents:  ## good place for tqdm
            os.chdir(start_dir)
        
            ## create solvent directory
            solvdir = f"{workdir}/{solvent}/fep-vdw"
            os.makedirs(solvdir, exist_ok=True)
        
            # Run Fep
            os.chdir(solvdir)
            for state in states:
                processes, offset = check_processes(processes,offset,ncores=36)
                proc = Process(target=staterunmd, args=[resname, state, offset])
                proc.start()
                processes.append(proc)
                offset += 1
            os.chdir(start_dir)
    ## Final process cleanup
    for proc in processes:
        proc.join()



####
#### Functions to analyse the partition simulations
####

# Function to list all xvg files recursively, ignoring directories starting with .
def list_xvg_files(directory):
    xvg_files = []
    for root, dirs, files in os.walk(directory):
        # Filter out directories starting with .
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith('.xvg'):
                xvg_files.append(os.path.join(root, file))
    return xvg_files

def TIRoutine(T=298,cutoff=5000, states=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):

    ### Sort the dhdl files
    xvg_files = list_xvg_files('.')
    for file in xvg_files:
        state = file[10:-8] #get state name 
        if not os.path.exists(f"./dhdl-{state}.xvg"):
            os.symlink(file, f"./dhdl-{state}.xvg")
    
    ### Load them all into alchemlyb
    xvg_links = [f"dhdl-{state}.xvg" for state in states]
    dhdl = alchemlyb.concat([extract_dHdl(xvg, T=T)[cutoff:] for xvg in xvg_links])
    
    ### Run TI
    resTI = TI().fit(dhdl)
    solvationdG = resTI.delta_f_.loc[0.00, 1.00] * (0.008314*298)  ## To convert KbT to kJ/mol at 298K
    error = resTI.d_delta_f_.loc[0.00, 1.00] * (0.008314*298)

    return solvationdG, error

def process_partition(workdir, solvents=['water','octanol-water_74-26','hexadecane','chloroform'],
                     reps=[1,2,3]):
    report=[]
    overall=[]
    main_dir = os.getcwd()
    os.chdir(workdir)
    report.append(f'Molecule: {workdir}\n')
    for rep in reps:
        dGs = []
        for solvent in solvents:
            solvdir = f"{workdir}/{rep}/{solvent}"
            os.chdir(solvdir)
            # solvationdG, error = TIRoutine()
            dGs.append(TIRoutine())
        
        out=[]
        report.append(f"Rep #{rep}\n")
        for idx, solvent in enumerate(solvents[1:]):
            partition_dG     = dGs[idx+1][0] - dGs[0][0]
            partition_dG_err = dGs[idx+1][1] + dGs[0][1]
            partitionlog     = partition_dG / (2.302585*0.008314*298)
            partitionlog_err = partition_dG_err / (2.302585*0.008314*298)
            report.append(f" {solvent}/water partition: {partitionlog}+-{partitionlog_err} LogP units\n")
            report.append(f" {solvent}/water partition: {partition_dG}+-{partition_dG_err} kJ/mol units\n")
            out.append([partition_dG,partition_dG_err,partitionlog,partitionlog_err])
        
        overall.append(out)
    
    rep_avg=[]
    report.append(f"Average\n")
    for idx, solvent in enumerate(solvents[1:]):
        partition_dG     = np.mean(np.array(overall)[:][:,idx,0])
        partition_dG_err = np.std(np.array(overall)[:][:,idx,0])
        partitionlog     = np.mean(np.array(overall)[:][:,idx,2])
        partitionlog_err = np.std(np.array(overall)[:][:,idx,2])
    
        rep_avg.append([partition_dG, partition_dG_err, partitionlog, partitionlog_err])  ##logs
        report.append(f" {solvent}/water partition: {partitionlog}+-{partitionlog_err} LogP units\n")
        report.append(f" {solvent}/water partition: {partition_dG}+-{partition_dG_err} kJ/mol units\n")
    
    with open(f'{workdir}/results.txt', 'w+') as topout:
        for line in report:
            topout.write(line)
    rep_avg=np.array(rep_avg)
    np.save(f'{workdir}/partition_avg.npy',rep_avg)
    os.chdir(main_dir)
