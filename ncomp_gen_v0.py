# Iterate through options lists
opt_dict = {"na":narrow_options,"br":broad_options,"abs":absorp_options}
ncomp_dict = {}
for line_type in opt_dict:
    ncomp = opt_dict[line_type]["ncomp"]
    if ncomp>0:    
        for n in np.arange(ncomp-1)+2:
            if ("NCOMP_%d" % n) not in ncomp_dict:
                ncomp_dict["NCOMP_%d" % n] = {}
    #         print(n)
            # Create a dictionary for each additional set of components
            for line in line_list: # line is the "parent" line
                if line_list[line]["line_type"]==line_type:
                    ncomp_dict["NCOMP_%d" % n][line+"_%d" % n] = {} 
        #             print(line_list[line])
        #             print(line)
                    # Create a new child line based on the parent line
#                     ncomp_dict["NCOMP_%d" % n] = {}
                    for hpar in line_list[line]:
                        # First non-fittable hyperparameters (center, line_type, line_profile)
                        if hpar=="center":
                            ncomp_dict["NCOMP_%d" % n][line+"_%d" % n][hpar] = line_list[line]["center"]
                        if hpar=="line_type":
                            ncomp_dict["NCOMP_%d" % n][line+"_%d" % n][hpar] = line_list[line]["line_type"]
                        if hpar=="line_profile":
                            ncomp_dict["NCOMP_%d" % n][line+"_%d" % n][hpar] = line_list[line]["line_profile"]
                        # Now fittable hyperparameters (amp, disp, voff, h3, shape, etc.)
                        # Parameters that are free in the parent will become free in the child. 
                        # Parameters that are tied in the parenter will become tied to their respective child component parameters.
                        if (hpar in ["amp","disp","voff","h3","h4","h5","h6","h7","h8","h9","h10","shape"]) and (line_list[line][hpar]=="free"):
                            ncomp_dict["NCOMP_%d" % n][line+"_%d" % n][hpar] = line_list[line][hpar]
                        elif (hpar in ["amp","disp","voff","h3","h4","h5","h6","h7","h8","h9","h10","shape"]) and (line_list[line][hpar]!="free"):
        #                     print(hpar,line_list[line][hpar])
        #                     print(line_list.keys())
                            for key in line_list.keys():
                                if key in line_list[line][hpar]:
                                    new_hpar = line_list[line][hpar].replace(key,key+"_%d" % n)
                                    ncomp_dict["NCOMP_%d" % n][line+"_%d" % n][hpar] = new_hpar

new_line_list = copy.deepcopy(line_list)             
for n in ncomp_dict:
    for line in ncomp_dict[n]:
        new_line_list[line] = ncomp_dict[n][line]