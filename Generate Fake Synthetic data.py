import numpy as np
import statsmodels.api as sm
import pandas as pd

# Set the seed for reproducibility
np.random.seed(45)

# No. of samples
N_SAMPLES = 10000

# Generate X
age = np.random.randint(18, 65, N_SAMPLES)
re19 = np.random.randint(20000, 60000, N_SAMPLES)
hisp = np.random.choice(a=[0,1], p=[0.7,0.3], size=N_SAMPLES)
black = np.random.choice(a=[0,1], p=[0.7,0.3],size=N_SAMPLES)
re20 = re19 + 1000 * np.random.randn(N_SAMPLES)
married = ((1 / (1 + np.exp(- (3 - (age*0.05) - (re20 * 0.00002) + np.random.randn(N_SAMPLES)))))>0.5).astype(int)
educ = -0.2*hisp + np.random.randint(7, 14, N_SAMPLES)
nodegr = ((1 / (1 + np.exp(- (1 - (age*0.05) - (educ*0.05) + np.random.randn(N_SAMPLES)))))>0.5).astype(int)
treat = ((1/(1+np.exp(nodegr*0.4 + np.random.randn(N_SAMPLES))))>0.5).astype(int)

# Compute Y
re22 = 54 * age -115*nodegr + 2000*treat + 396*educ - 800*black + 4000*np.random.randn(N_SAMPLES) + 35550
re22=re22.round(0).astype(int)

data = pd.DataFrame({
    "treat":treat,
    "age" : age,
    "hisp" : hisp,
    "black":black,
    "married":married,
    "educ":educ,
    "nodegr":nodegr,
    "re19" :re19,
    "re20":re20,
    "re22":re22
})

data.to_csv("data.csv",index=False)