{
'ofr': [1, 12, 11], # range for relative offsets
'mrf': [.5, 1.5, 10], # range for the first element of the mean
'mr': [-1.5, 1.5, 30], # range for other elements of the mean
'sr': [-.25, .25, 20], # rango for off-diagonal elements of the covariance
'srd': [.01, 1, 20], # range for digonal elements of the covariance
'base': {'threads': 0, 'size': 300, 'time': 5.0}, #optimizer base parameters
'mod': {'nd': 3, 'slen': 800, 'compress': 'istac', 'clevel': 4}, #parameters for the acellcross model
'ga': {'crossover': .9, 'mutation': .05, 'transposition': .001, 'minprop': 0} #parameters for the GA
}
