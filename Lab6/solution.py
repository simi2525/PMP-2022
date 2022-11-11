import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":

    data = pd.read_csv('data.csv')

    educ_cat = data['educ_cat'].values
    momage = data['momage'].values
    ppvt = data['ppvt'].values

    # Ex1
    fig, axes = plt.subplots(1, 2, sharex=False, figsize=(10, 4))
    axes[0].scatter(educ_cat, ppvt, alpha=0.6)
    axes[1].scatter(momage, ppvt, alpha=0.6)
    axes[0].set_ylabel("Cog. score")
    axes[0].set_xlabel("Level of education")
    axes[1].set_xlabel("Age of mother")
    plt.savefig('cog_score_corelations.png')
    plt.close()

    cog_score_mom_model = pm.Model()

    # Ex2 / Ex4
    with cog_score_mom_model:
        a = pm.Normal('a', mu=0, sd=10)
        
        # bEdu = pm.Normal('bEdu', mu=0, sd=10)
        bMom = pm.Normal('bMom', mu=0, sd=10)
        
        sigma = pm.HalfNormal('sigma', sd=1)

        # mu = pm.Deterministic('mu',a + bEdu * educ_cat)
        mu = pm.Deterministic('mu',a + bMom * momage)
        
        ppvt_like = pm.Normal('ppvt_like', mu=mu, sd=sigma, observed=ppvt)

        trace = pm.sample(20000, tune=20000, cores=4)

    # Ex3 / Ex4
    a_mean = trace['a'].mean().item()
    
    
    # bEdu_mean = trace['bEdu'].mean().item()
    # plt.plot(educ_cat, ppvt, 'o', alpha=0.6)
    # plt.plot(educ_cat, a_mean + bEdu_mean * educ_cat, 'r')
    # plt.xlabel('Education level')
    # plt.ylabel('Cog. score', rotation=0)
    # plt.savefig('regression_line_mom_edu.png')
    
    bMom_mean = trace['bMom'].mean().item()
    plt.plot(momage, ppvt, 'o', alpha=0.6)
    plt.plot(momage, a_mean + bMom_mean * momage, 'r')
    plt.xlabel('Mom age')
    plt.ylabel('Cog. score', rotation=0)
    plt.savefig('regression_line_mom_age.png')
    
    plt.close()
    
    # Bonus
    
    ppc = pm.sample_posterior_predictive(trace, samples=100, model=cog_score_mom_model)
    
    # plt.plot(educ_cat, a_mean + bEdu_mean * educ_cat, 'r')
    # sig = az.plot_hdi(educ_cat, ppc['ppvt_like'], hdi_prob=0.97, color='k')
    # plt.xlabel('Education level')
    # plt.ylabel('Cog. score', rotation=0)
    # plt.savefig('bayesian_regression_line_mom_edu.png')
    
    
    plt.plot(momage, a_mean + bMom_mean * momage, 'r')
    sig = az.plot_hdi(momage, ppc['ppvt_like'], hdi_prob=0.97, color='k')
    plt.xlabel('Mom age')
    plt.ylabel('Cog. score', rotation=0)
    plt.savefig('bayesian_regression_line_mom_age.png')
