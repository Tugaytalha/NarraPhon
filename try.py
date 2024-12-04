
import os

pptx_inp = "./logs/as.pptx"
output_dir = "./sa"
pptx_inp = os.path.join(output_dir, os.path.basename(pptx_inp))

print(pptx_inp)