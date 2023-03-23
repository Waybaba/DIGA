# read from val.txt, randomly select 100 lines and write to val_small.txt

import random
if __name__ == '__main__':
	with open('val.txt', 'r') as f:
		lines = f.readlines()
	with open('small.txt', 'w') as f:
		for i in range(100):
			f.write(random.choice(lines))