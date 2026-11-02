import pickle, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from paths import DATA, DERIVED, MODELS, FIGURES
D = DERIVED
g=pd.read_csv(D+"converged_with_weights.csv")
X=g[["P_N2","P_CO2","P_CH4","T_surf_climate"]]; Y=g[["chi2","reduced_chi2","n_transits_for_detection"]]
det=RandomForestRegressor(500,random_state=42,n_jobs=-1).fit(X.to_numpy(float),Y.to_numpy(float))
clim=RandomForestRegressor(500,random_state=42,n_jobs=-1).fit(X.iloc[:,:3].to_numpy(float),X.T_surf_climate.to_numpy(float))
rng={c:{"min":float(X[c].min()),"max":float(X[c].max())} for c in X}
pickle.dump({"climate_model":clim,"detection_model":det,"climate_feature_columns":["P_N2","P_CO2","P_CH4"],
 "detection_feature_columns":list(X.columns),"target_columns":list(Y.columns),"training_ranges":rng,
 "detection_metrics":{},"n_training_models":len(g)},open(MODELS+"TRAPPIST1e_Climate_Detection_AI_Combined_v2_all1856.pkl","wb"))
print("saved")
