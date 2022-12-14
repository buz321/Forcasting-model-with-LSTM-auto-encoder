# Forcasting-model-with-LSTM-auto-encoder

Forcasting stock return with deep learning methods based on the historical data

# Result Analysis
# Result 1

Table 1: Sequence Length (Length 4 is the benchmark)

![image](https://user-images.githubusercontent.com/107760647/189462943-b18837f8-2dc5-43ea-8cee-d8bee6a3258c.png)

Table 2: Latent Feature (128 is the benchmark)

![image](https://user-images.githubusercontent.com/107760647/189462963-b1a6fb72-162d-4b18-ae67-8c530d2a2df9.png)

After changing sequence length and latent feature, we could find the optimal parameter. However, it seems like there is no trend when changing. 

# Result 2

<Graph 1: Importance Feature>
 ![image](https://user-images.githubusercontent.com/107760647/189463006-bc8354cd-5051-42f4-a8b9-74d7f24e5575.png)

We could identify the variables that contribute the most to the dependent variable.

# Result 3

Graph 3: Linear Tree splits
 ![image](https://user-images.githubusercontent.com/107760647/189463140-6555fd0c-7fd7-4de6-8b10-a5040c363ff5.png)

The original model's mean absolute error (MAE) is 0.1233, which is higher than the linear tree model (MAE: 0.1213).
