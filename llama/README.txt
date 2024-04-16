Designed for running locally on Mac
Please install MLX first

1. Code
evo_llama.py: learning a simple test over a random number of feedbacks and a random scale.
    * edit TODO1 (training questions), TODO2 (test question), TODO3 (training loss), TODO4 (test loss, should be equal to TODO3 usually)

Want to understand what the code is doing ? << diff llama.py evo_llama.py >>

2. Scripts for running
meta_evo_lama.sh: launches a loop of evo_lama.sh
evo_lama.sh: launches evo_llama.py

3. Scripts for plotting
meta_view_stats.sh : plots train and test curves
view_stats.sh
