import sys
sys.path.append("debias")

from generator.generate import generate_images
from generator.generate_dgm import generate_images_dgm
from generator.dir_build import build_directory

from align import process_all_subdirs

from discriminator.discriminator import discriminator

from executor.executor import execute
