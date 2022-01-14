import os, glob	# to set directory
import os.path

targetdir0 = r"./opcode/0/"	# ransomwares
targetdir1 = r"./opcode/1/"	# good files

files0 = os.listdir(targetdir0)	# file name list of 0
files1 = os.listdir(targetdir1)	# file name list of 1

instrlist = []
outlist = ''

def update_instr_list(d, file_list):
# param - d: directory, file_list - list of file names in d
	for i in file_list:
		f = open(d+i, 'r')
		lines = f.readlines()
		for j in lines:
			if j not in instrlist:
				instrlist.append(j)
		f.close()
		
update_instr_list(targetdir0, files0)
update_instr_list(targetdir1, files1)

for i in instrlist:
	outlist += str(i)
	
f = open('instrlist.txt', 'w')
f.write(outlist)
f.close()
