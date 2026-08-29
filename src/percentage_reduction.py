def percentage_reduction(original, new):
    return (new - original) / original * 100

if __name__ == "__main__":
    transformer_params = 16.7780 
    evo_params = 16.8359
    reduction = percentage_reduction(transformer_params, evo_params)
    print(f"Reduction: {reduction:.2f}%")
    exit()
    
    