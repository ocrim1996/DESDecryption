import requests
import os.path
import re

def generate(filename, output, max_lines, n=8):
	i = 0
	regexp = re.compile('^([a-zA-Z0-9]{' + str(n) + '})$')
	with open(output, 'a') as out:
		out.truncate(0) # empty file before starting
		with open(filename) as dict:
			for line in dict:
				line = line.rstrip()
				if regexp.search(line):
					i += 1
					out.write(line + '\n')
					if i > max_lines: break
		out.truncate(out.tell()-1) # strip last newline

if not os.path.isfile("./xato-net-10-million-passwords.txt"):
	# Download master file (~50MB)
	print('downloading xato-net-10-million-passwords.txt...')
	r = requests.get('https://github.com/danielmiessler/SecLists/blob/master/Passwords/xato-net-10-million-passwords.txt?raw=true')
	with open("./xato-net-10-million-passwords.txt", "wb") as file:
		file.write(r.content)

# Extract dictionaries
for n_lines in [100, 1000, 10000, 100000, 100000, 1000000, 1500000]:
	name = "dict_" + str(n_lines) + ".txt"
	print('generating ' + name + '...')
	generate("xato-net-10-million-passwords.txt", name, n_lines)

print('done!')
