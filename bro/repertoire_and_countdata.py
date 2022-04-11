import numpy as np 
import pandas as pd
from scipy.sparse import diags
from scipy.special import gammaln, factorial
from scipy.stats import hypergeom

from bro.utilityfunctions import uniform_distribution_pdf, normalize_pdf

class DistributionComputations:
    """Class that implements the pieces that are needed to compute the joint posterior distribution of s, Ra, and Rb given count data.
    """
    def log_multinomial_coefficient(self, N, m):
        """
        Calculates logarithm of multinomial coefficient for N and m=(m1,...,mK), i.e.,
        log(N!/(m1! * m2! *** mK!))

        :param int N: integer in numerator of multinomial coefficient
        :param list(int) m: list of integers that should sum to N
        :return float: log of multinomial coefficient
        """
        assert np.sum(m)==N, 'sum(m)={} needs to be N={}'.format(np.sum(m), N)
        return gammaln(N+1) - np.sum(gammaln(m+1))

    def log_probability_of_count_data_given_repertoire_size(self, count_data, R):
        """Calculates logarithm of p(C|R) where C is count data and R is repertoire size.

        :param list count_data: count data as list of sampled elements    
        :param int R: repertoire size
        :return float: logarithm of p(C|R)
        """
        N = len(count_data)
        g, c = np.unique(count_data, return_counts=True)
        observed_R = len(g)
        if R<observed_R:
            return np.log(0)
        else:
            c = np.array(c.tolist() + [0]*int(R-observed_R))
            assert len(c)==R, f'R={R}, len(c)={len(c)}'
            uniqCts, freqCts = np.unique(c, return_counts=True)

            x = self.log_multinomial_coefficient(R, freqCts)
            y = self.log_multinomial_coefficient(N, c)
            z = N*np.log(R)
            return x+y-z

    def pdf_of_repertoire_size_given_count_data(self, count_data, R_prior):
        """Calculates p(R|C).

        :param list count_data: list of sampled elements
        :param np.array R_prior: 2xN prior on the repertoire size
        :return np.array: 2xN posterior distribution p(R|C) with the same support as the prior.
        """
        pdf = []
        for R, prob in R_prior:
            pdf.append([R, self.log_probability_of_count_data_given_repertoire_size(count_data, R) + np.log(prob)])
        pdf = np.array(pdf)
        pdf[:,1] = np.exp(pdf[:,1])
        assert pdf[:,1].sum()!=0, 'pdf sums to 0'
        pdf = normalize_pdf(pdf)
        return pdf


    def pdf_of_s_given_na_nb_nab_Ra_Rb(self, nab, na, nb, Ra, Rb):
        """Given data D=(nab,na,nb,Ra,Rb), calculates pdf of the overlap given the data p(s|D). 
        For reference: hypergeom.pmf(special_drawn, total, special, draws, loc=0)
        
        :param int nab: empirical overlap
        :param int na: number of unique items sampled from set A
        :param int nb: number of unique items sampled from set A
        :param int Ra: size of set A
        :param int Rb: size of set B
        :return np.array: 2xN array representing p(s|D) in [[s,p(s|D)],...] form
        """
        # find smaller index
        R, n = [Ra, Rb], [na, nb]
        assert ((na<=Ra) and (nb<=Rb) and nab<=min(R) and (nab<=na) and (nab<=nb)), 'values dont make sense: n={}, R={}, nab={}'.format(n, R, nab)
        smaller_idx = np.argmin(R)
        larger_idx = (smaller_idx+1)%2
        
        # s given Ra and Rb prior
        s_given_Ra_Rb = uniform_distribution_pdf(np.arange(0,min([Ra,Rb])+1))
        
        # calculating log(p(nab|na,nb,s,Ra,Rb)) for all possible s values \in [nab, min(Ra,Rb)]
        log_nab_given_na_nb_s_Ra_Rb = []
        overlap_values = np.arange(nab, int(min([Ra,Rb])+1))
        for i, s in enumerate(overlap_values):
            # p_sa is the probability that we'd get sa from the overlap (s), just in na draws of a
            prob_sa = hypergeom.pmf(np.arange(R[smaller_idx]+1),R[smaller_idx],s,n[smaller_idx])
            # p_nab_given_sa is the probability of getting that nab, given sa
            prob_nab_given_sa = hypergeom.pmf(nab,R[larger_idx],np.arange(R[smaller_idx]+1),n[larger_idx])
            
            log_nab_given_na_nb_s_Ra_Rb.append(np.log(np.dot(prob_sa,prob_nab_given_sa)))
        
        # calculating p(s|na,nb,nab,Ra,Rb) \propto p(nab|na,nb,s,Ra,Rb)p(s|Ra,Rb)
        log_s_given_na_nb_nab_Ra_Rb = np.array(log_nab_given_na_nb_s_Ra_Rb) + np.log(s_given_Ra_Rb[s_given_Ra_Rb[:,0]>=nab][:,1])
        s_given_na_nb_nab_Ra_Rb = np.exp(log_s_given_na_nb_nab_Ra_Rb)
        
        # normalizing pdf and making it have format [[val, p(val)],...]
        s_given_na_nb_nab_Ra_Rb = s_given_na_nb_nab_Ra_Rb/np.sum(s_given_na_nb_nab_Ra_Rb)
        s_given_na_nb_nab_Ra_Rb = np.array([[s,s_given_na_nb_nab_Ra_Rb[i]] for i,s in enumerate(overlap_values)])

        return s_given_na_nb_nab_Ra_Rb

    def calculate_joint_pdf_from_count_data(self, count_data, Ra_prior, Rb_prior):
        """Compute the joint posterior distribution of (s, Ra, Rb) given count data and priors on the
        repertoire sizes.

        :param CountDataPair count_data: object that contains the count data from the two sets 
        :param np.array Ra_prior: 2xN prior on the repertoire size Ra
        :param np.array Rb_prior: 2xN prior on the repertoire size Rb
        :return pd.DataFrame: dataframe representing the posterior distribution of (s, Ra, Rb) given
            the count data
        """
        nab, na, nb = count_data.nab, count_data.na, count_data.nb
        Ra_dist = self.pdf_of_repertoire_size_given_count_data(count_data.count_dataA, Ra_prior)
        Rb_dist = self.pdf_of_repertoire_size_given_count_data(count_data.count_dataB, Rb_prior)

        p_s = {}
        for Ra in Ra_dist[Ra_dist[:,1]>0][:,0]:
            for Rb in Rb_dist[Rb_dist[:,1]>0][:,0]:
                p_s[f'{int(Ra)},{int(Rb)}'] = self.pdf_of_s_given_na_nb_nab_Ra_Rb(nab,na,nb,Ra,Rb)
        
        p_Ra = pd.Series(Ra_dist[:,1],index=[int(x) for x in Ra_dist[:,0]])
        p_Rb = pd.Series(Rb_dist[:,1],index=[int(x) for x in Rb_dist[:,0]])

        for key in p_s.keys():
            Ra, Rb = [int(x) for x in key.split(',')]
            p_s[key][:,1] = np.exp(np.log(p_s[key][:,1])+np.log(p_Ra[Ra])+np.log(p_Rb[Rb]))

        joint_pdf = []
        for key in p_s.keys():
            Ra, Rb = [int(x) for x in key.split(',')]
            joint_pdf.extend([[int(s),Ra,Rb,prob] for s, prob in p_s[key]])
        joint_pdf = pd.DataFrame(joint_pdf, columns=['s','Ra','Rb','p(s,Ra,Rb)'])

        return joint_pdf

class CountDataPair:
    """Helper class to collect count data for two repertoires."""
    def __init__(self, count_dataA, count_dataB):
        self.count_dataA = count_dataA
        self.count_dataB = count_dataB

        self.ma = len(self.count_dataA)
        self.mb = len(self.count_dataB)
        self.na = len(np.unique(self.count_dataA))
        self.nb = len(np.unique(self.count_dataB))
        self.nab = len(np.intersect1d(self.count_dataA, self.count_dataB))
         
class GenerateCountData:
    """Class for generating synthetic count data.
    """
    def __init__(self, s, Ra, Rb, ma, mb):
        """Constructor

        :param int s: repertoire overlap
        :param int Ra: repertoire A size
        :param int Rb: repertoire B size
        :param int ma: number of items sampled from A
        :param int mb: number of items sampled from B
        """
        assert ((Ra-int(Ra)==0) and (Rb-int(Rb)==0) and (s-int(s)==0) and (ma-int(ma)==0) and (mb-int(mb)==0)), 'variables should be integers! Ra={},Rb={},s={},ma={},mb={}'.format(Ra,Rb,s,ma,mb)
        self.s = int(s)
        self.Ra = int(Ra)
        self.Rb = int(Rb)
        self.ma = int(ma)
        self.mb = int(mb)
        self.generate_count_data()
    
    def generate_gene_pools(self):
        """
        Create sets of genes for two parasites with repertoire sizes Ra and Rb, respectively, and overlap s. The genes IDed by number with the first s of each set being the same numbers. genesetA=[0,1,...,Ra-1] and genesetB=[0,1,...,s-1,Ra,Ra+1,...,Ra+(Rb-s)-1]. 

        E.g., if Ra=5, Rb=6, s=3, then genesetA=[0,1,2,3,4] and genesetB=[0,1,2,5,6,7]

        :param int s: repertoire overlap
        :param int Ra: parasite A repertoire size
        :param int Rb: parasite B repertoire size
        :return tuple: 2-tuple consisting of genesetA and genesetB (see docstring)
        """
        gene_pool_A = np.arange(0,self.Ra)
        gene_pool_B = np.array(list(range(0,self.s)) + list(range(self.Ra, self.Ra+(self.Rb-self.s))))
        assert len(gene_pool_A)==self.Ra and len(gene_pool_B)==self.Rb, ''

        return (gene_pool_A, gene_pool_B)

    def sample_from_gene_pool(self, m, gene_pool):
        """Draws samples with replacement for a gene pool. The gene pool is type returned by generate_genesets_given_s_and_R. 
        Returns the genes drawn.
        """
        count_data = np.random.choice(gene_pool, size=m, replace=True, p=np.repeat(1/len(gene_pool), len(gene_pool)))
        return count_data
        

    def generate_count_data(self):
        """Given a repertoire overlap, repertoire sizes, and number of samples per repertoire to draw. Returns
        the genes drawn from parasite A and genes drawn from parasite B
        """
        gene_pool_A, gene_pool_B = self.generate_gene_pools()
        count_dataA = self.sample_from_gene_pool(self.ma, gene_pool_A)
        count_dataB = self.sample_from_gene_pool(self.mb, gene_pool_B)
        self.count_data = CountDataPair(count_dataA, count_dataB)

class ProcessJointPdfs:
    """Utilities to process joint posterior distribution of (s, Ra, Rb) given count data.
    """
    def compute_marginal_distribution(self, variable, joint_pdf):
        """Given p(s,Ra,Rb|Ca,Cb) computes p(var|Ca,Cb) by summing over other two
        variables. var \in {s,Ra,Rb}.
        """
        marginal = joint_pdf.groupby([variable], as_index=False)['p(s,Ra,Rb)'].sum()
        return marginal

    def confidence_interval_from_pdf_for_integer_valued_rv(self, pdf, prob=0.95):
        """Given the pdf for an integer valued random variable, compute X% confidence
        interval. 
        """
        res = []
        for i in range(pdf.shape[0]):
            if pdf[i:,1].cumsum()[-1]<prob:
                break
            j = np.abs(pdf[i:,1].cumsum()-prob).argmin()+i

            res.append([i,j, pdf[i:j+1,1].sum()])
        res = np.array(res)
        lower, upper, ci_prob = res[np.abs(res[:,2]-prob).argmin()]
        lower, upper = int(lower), int(upper)
        
        mean = np.sum(pdf[:,0]*pdf[:,1])

        return {'low': pdf[lower,0], 'mean': mean, 'high': pdf[upper,0], 'ci_prob': ci_prob}
    
    def compute_marginal_and_ci_from_joint_pdf(self, variable, joint_pdf, prob=0.95):
        """
        """
        results = {}
        # for var in ['s','Ra','Rb']:
            # print(f'var = {var}')
        marginal_distribution = self.compute_marginal_distribution(variable,joint_pdf).to_numpy()
        results = {'marginal': marginal_distribution, 
                   'ci': self.confidence_interval_from_pdf_for_integer_valued_rv(marginal_distribution, prob)}
        return results


    def sorensen_dice_coefficients(self, joint_pdf, s, Ra, Rb, na, nb, nab):
        """Compute the true, empirical, and Bayesian Sorensen-Dice coefficients.
        true = 2*s/(Ra+Rb), empirical = 2*nab/(na+nb),
        Bayesian = \sum_{Ra,Rb,s} 2*s/(Ra+Rb) * p(s,Ra,Rb|Ca,Cb)
        """
        true_sorensen_dice = 2*s/(Ra+Rb)
        empirical_sorensen_dice = 2*nab/(na+nb)

        tmp = joint_pdf.apply(lambda x: 2*x['s']/(x['Ra']+x['Rb']), axis=1)
        bayesian_sorensen_dice = (tmp*joint_pdf['p(s,Ra,Rb)']).sum()

        return {'true_sd': true_sorensen_dice, 'empirical_sd': empirical_sorensen_dice,
                'bayesian_sd': bayesian_sorensen_dice}

def generate_count_data_and_compute_joint_pdf(s, Ra, Rb, ma, mb, Ra_prior, Rb_prior):
    """Helper function to generate synthetic count data and then compute the joint posterior 
    distribution of (s, Ra, Rb) given the count data and repertoire priors.

    :param int s: size of set overlap
    :param int Ra: size of set A
    :param int Rb: size of set B
    :param int ma: number of items sampled from set A
    :param int mb: number of items sampled from set A
    :param np.array Ra_prior: 2xN prior on the repertoire size Ra
    :param np.array Rb_prior: 2xN prior on the repertoire size Rb
    :return dict: dictionary of count data and joint pdf
    """
    print('Generating count data')
    data = GenerateCountData(s, Ra, Rb, ma, mb)
    count_data = data.count_data

    print('Calculating joint pdf...')
    computation = DistributionComputations()
    joint_pdf = computation.calculate_joint_pdf_from_count_data(count_data, Ra_prior, Rb_prior)
    print('Done calculating joint pdf...')

    results = {
        # 's': s, 'Ra': Ra, 'Rb': Rb,
        # 'Ra_prior': Ra_prior, 'Rb_prior': Rb_prior,
        'count_data': count_data,
        'joint_pdf': joint_pdf
    }

    return results
    
def process_file(file_path, save_path):
    variables = ['s','Ra','Rb']
    row = {}
    row['path'] = file_path

    d = unpickle_object(file_path)
    joint_pdf = pd.DataFrame(d['joint_pdf'])

    for var in variables:
        marginal_dist_and_ci = processor.compute_marginal_and_ci_from_joint_pdf(var,joint_pdf,prob)
        tmp = {f'{var}_{key}': val for key, val in marginal_dist_and_ci['ci'].items()}
        row.update(tmp)

    row.update(processor.sorensen_dice_coefficients(joint_pdf,d['s'],d['Ra'],d['Rb'],d['na'],d['nb'],d['nab']))
    row.update({key: d[key] for key in keys_to_keep})
    
    pickle_object(save_path, row)