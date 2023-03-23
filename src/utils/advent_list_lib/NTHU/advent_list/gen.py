"""
generage {set}.txt from test.txt

examples in test.txt
	Rio/pano_00239_2_0.png
	Rio/pano_00358_5_180.png
	Rio/pano_00370_0_180.png
	Rio/pano_00373_0_180.png
	Taipei/pano_01728_0_180.png
	Tokyo/pano_00002_2_0.png
	Tokyo/pano_00022_2_0.png
	Rio/pano_00631_0_0.png
	Rio/pano_03575_2_0.jpg
	Rio/pano_03584_0_0.jpg
"""
import os, random

def gen_city_set(set="Rio"):
	"""
	just select the line with city name
	remove before start if target file exists
	"""
	os.system("rm %s.txt" % set)
	lines = []
	with open("test.txt", "r") as f:
		with open(set + ".txt", "w") as f2:
			for line in f:
				if set in line:
					# if not end with newline char, add it
					if line[-1] != "\n":
						line = line + "\n"
					lines.append(line)
			# remove the last newline char
			lines[-1] = lines[-1].replace("\n", "")
			for line in lines:
				f2.write(line)

def gen_random_order_test_set():
	"""
	read test.txt, and write to test_random_order.txt
	remove before start if target file exists
	note that the last line of test.txt has no newline char
	"""
	os.system("rm test_random_order.txt")
	with open("test.txt", "r") as f:
		lines = f.readlines()
		lines[-1] = lines[-1] + "\n"
		random.shuffle(lines)
		lines[-1] = lines[-1].replace("\n", "")
		with open("test_random_order.txt", "w") as f2:
			for line in lines:
				f2.write(line)


if __name__ == "__main__":
	# gen all city set (Rio, Taipei, Tokyo, Rome)
	gen_city_set("Rio")
	gen_city_set("Taipei")
	gen_city_set("Tokyo")
	gen_city_set("Rome")
	# gen random order test set
	gen_random_order_test_set()
