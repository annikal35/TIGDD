class SubDue:
    SubDueFolder ="subdue"
    graphFolder = "subdue/graphs"
    subgraph = "subgraph.g"
    graphFile = "G.g"
    minsize = 2
    maxsize = 10
    nsubs = 5
    beam = 4
    subdueCommand = "bin/subdue "+ " -minsize "+ str(minsize) +" -maxsize " + str(maxsize) + " -beam " + str(beam) +" -nsubs "+ str(nsubs)
    #subdueCommand = "bin/subdue "  + " -nsubs " + str(nsubs)


class Experiment:
    iterations = 100
    param_n = 10        #Number of subgraph
    param_w = 4     #Window Size
    param = [param_n, param_w]


class RULSIF:
    n = 50
    k = 10
    alpha = 0.1
    k_fold = 5
    th = 3.4

