'''
words
'''

# packages


# scripts


def build_weights(method_dict):
    '''
    expecting a dict like this:
     method_dict = { method1: weight1,
                     method2: weight2,}
    '''
    prob_remaining = 1.0
    methods = []
    weights = [0] * len(method_dict)
    equally_distribute = []
    for methtype, prob in method_dict.items():
        methods.append(methtype)
    	if prob <= 0:
    		weights[i] = 0
    		continue
		elif prob < 1:
			prob_remaining -= prob
			if prob_remaining < 0:
                print("UserInputError: current sum of prob/weights for %s is > 1" % methtype)
                exit()
            else:
            	weights[i] = prob
    	else:
    		# user wants this prob to be equally distributed with whatever is left
            equally_distribute.append(i)
    # we looped through all methods, now equally distribute the remaining amount
    if len(equally_distribute) > 0:
        eq_weight = round(prob_remaining/len(equally_distribute), 4)
        for i in equally_distribute:
            weights[i] = eq_weight
    # now clean up any rounding errors by appending any remainder to the last method
    remainder = 1 - sum(weights)
    if remainder > .01:
        print("UserInputError: total sum of prob/weights for %s is < .99" % method)
        exit()
    else:
        weights[-1] += remainder
    return methods, weights