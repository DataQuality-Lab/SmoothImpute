import numpy as np
import pandas as pd
import torch
from scipy import optimize

import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import KeyedVectors
import openai

def binary_sampler(p, rows, cols):
    '''Sample binary random variables.
    
    Args:
        - p: probability of 1
        - rows: the number of rows
        - cols: the number of columns
        
    Returns:
        - binary_random_matrix: generated binary random matrix.
    '''

    unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
    binary_random_matrix = 1*(unif_random_matrix < p)
    return binary_random_matrix

##### Missing At Random ######

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask

##### Missing not at random ######

def MNAR_mask_logistic(X, p, p_params =.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask

def MNAR_self_mask_logistic(X, p):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    mask = ber < ps if to_torch else ber < ps.numpy()

    return mask


def MNAR_mask_quantiles(X, p, q, p_params, cut='both', MCAR=False):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    q : float
        Quantile level at which the cuts should occur
    p_params : float
        Proportion of variables that will have missing values
    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.
        
    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
        
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """
    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1) ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == 'upper':
        quants = torch.quantile(X[:, idxs_na], 1-q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = torch.quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == 'both':
        u_quants = torch.quantile(X[:, idxs_na], 1-q, dim=0)
        l_quants = torch.quantile(X[:, idxs_na], q, dim=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
    ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na, dtype=X.dtype)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            try:
                intercepts[j] = optimize.bisect(f, -50, 50)
            except ValueError as e:
                print(f"Error in bisection method for self_mask at index {j}: {e}")
                intercepts[j] = float('nan')
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            try:
                intercepts[j] = optimize.bisect(f, -50, 50)
            except ValueError as e:
                print(f"Error in bisection method at index {j}: {e}")
                intercepts[j] = float('nan')
    return intercepts


def produce_NA(X, p_miss, mecha="MCAR", n_row=None, n_col=None, opt="quantile", p_obs=None, q=0.3):
    """
    Generates a mask for missing values in a dataset based on specified missing data mechanisms.

    Parameters:
    X (torch.Tensor): The input data tensor from which missing values will be generated.
    p_miss (float): The proportion of missing values to introduce.
    mecha (str): The mechanism for generating missing values. Options include:
        - "Random": Randomly assigns missing values.
        - "MCAR": Missing Completely At Random.
        - "MAR": Missing At Random, using the MAR_mask function.
        - "MNAR": Missing Not At Random, with options for quantile or self-masked methods.
    n_row (int, optional): The number of rows in the input data. Required for certain mechanisms.
    n_col (int, optional): The number of columns in the input data. Required for certain mechanisms.
    opt (str, optional): Additional option for MNAR mechanism, specifying the method to use.
    p_obs (float, optional): The proportion of observed values, used in MAR and MNAR mechanisms.
    q (float, optional): A quantile value used in the MNAR mechanism with quantile option.

    Returns:
    torch.Tensor: A tensor representing the mask for missing values, where 1 indicates observed values and 0 indicates missing values.
    
    Raises:
    ValueError: If the specified missing mechanism is not implemented.

    Example:
    >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> mask = produce_NA(X, p_miss=0.2, mecha="MCAR", n_row=2, n_col=2)
    """

    if n_row == None:
        n_row = X.shape[0]
    if n_col == None:
        n_col = X.shape[1]
    
    if p_miss == 0:
        mask = torch.ones((n_row, n_col), dtype=torch.float32) == 1
        return mask.view(-1)

    if mecha == "Random":
        unif_random_matrix = np.random.uniform(0., 1., size=X.shape[0])
        binary_random_matrix = 1 * (unif_random_matrix < (1 - p_miss))
        mask = torch.FloatTensor(binary_random_matrix) == 1
    elif mecha == "MCAR":
        unif_random_matrix = np.random.uniform(0., 1., size=[n_row, n_col])
        binary_random_matrix = 1 * (unif_random_matrix < (1 - p_miss))
        mask = torch.FloatTensor(binary_random_matrix) == 1
        # print(mask.shape)
    elif mecha == "MAR":
        # print(type(X))
        to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
        if not to_torch:
            X = X.astype(np.float32)
            X = torch.from_numpy(X)
        if p_obs == None:
            p_obs = 0.3*(1-p_miss)
        mask = MAR_mask(X.view(n_row, n_col), p_miss, p_obs).double()
        mask = mask == 0
    elif mecha == "MNAR" and opt == "quantile":
        if p_obs == None:
            p_obs = 0.3*(1-p_miss)
        to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
        if not to_torch:
            X = X.astype(np.float32)
            X = torch.from_numpy(X)
        mask = MNAR_mask_quantiles(X.view(n_row, n_col), p_miss, q, 1 - p_obs).double()
        mask = mask == 0
    elif mecha == "MNAR" and opt == "logistic":
        if p_obs == None:
            p_obs = 0.3*(1-p_miss)
        if torch.is_tensor(X):
            mask = MNAR_mask_logistic(X.double(), p_miss, p_obs).double()
        else:
            mask = MNAR_mask_logistic(torch.from_numpy(X).double(), p_miss, p_obs).double()
        mask = mask == 0
    elif mecha == "MNAR" and opt == "selfmasked":
        to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
        if not to_torch:
            X = X.astype(np.float32)
            X = torch.from_numpy(X)
        mask = MNAR_self_mask_logistic(X.view(n_row, n_col), p_miss).double()
        # mask = torch.FloatTensor(mask) == 1
        mask = mask == 0
    else:
        raise ValueError("Missing mechanism not implemented")
    return mask.view(-1)




# Generate vocabulary using the new API
def generate_vocabulary_with_llm(column_names, vocab_size=50):
    """
    Use the new OpenAI API to generate vocabulary for given column names.

    Args:
        column_names (list): List of column names.
        vocab_size (int): Number of vocabulary words per column.

    Returns:
        dict: A dictionary of generated vocabulary for each column.
    """
    # API key
    openai.api_key = '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'
    client = openai.OpenAI()
    prompt = (
        f"Generate a vocabulary of {vocab_size} meaningful words for the following column names in a table. "
        f"The vocabulary should be semantically relevant to the column names. "
        f"Here are the column names: {', '.join(column_names)}. \n"
        f"Return the vocabulary as a list of words for each column name."
    )

    # Request to generate vocabulary based on the column names
    # response = openai.Completion.create(
    #     model="gpt-4o-mini",  # Make sure to select the correct model
    #     prompt=prompt,
    #     max_tokens=300  # Limit the response size
    # )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    # Parse the generated text into a dictionary (adjust the format if necessary)
    generated_text = response.choices[0].message.content
    print(generated_text)
    
    # Parse the generated text into a dictionary
    vocab_dict = {}
    current_column = None
    words = []
    
    for line in generated_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.endswith(':'):
            # Save previous column's words if they exist
            if current_column is not None:
                vocab_dict[current_column] = words
            
            # Start new column
            current_column = line[:-1]
            words = []
        else:
            # Add words for current column
            words.extend([w.strip() for w in line.split(',') if w.strip()])
    
    # Add the last column
    if current_column is not None:
        vocab_dict[current_column] = words
        
    return vocab_dict

class Simulator:
    def __init__(self, mcar_p=0, mar_p=0, mnar_p=0):
        self.mcar_p = mcar_p
        self.mar_p = mar_p 
        self.mnar_p = mnar_p

    def simulate(self, data, simulate_data=False, distribution="normal", row_shape=5, column_shape=5, text_data=False, column_name = ["phone_number", "address", "restaurant_name"]):
        if simulate_data:
            return self.simulate_data(distribution, row_shape, column_shape, text_data, column_name)
        else:
            return self.simulate_mask(data)

    def simulate_data(self, distribution, row_shape, column_shape, text_data, column_name):
        """Generate synthetic datasets from different distributions"""
        
        # Normal distribution
        if distribution == "normal":
            normal_data = np.random.normal(loc=0, scale=1, size=(row_shape, column_shape))
            datasets = pd.DataFrame(normal_data)
        
        # Uniform distribution
        if distribution == "uniform":
            uniform_data = np.random.uniform(low=-1, high=1, size=(row_shape, column_shape))
            datasets = pd.DataFrame(uniform_data)
        
        # Exponential distribution
        if distribution == "exp":
            exp_data = np.random.exponential(scale=1.0, size=(row_shape, column_shape))
            datasets = pd.DataFrame(exp_data)
        
        # Chi-square distribution
        if distribution == "chi-square":
            chi_data = np.random.chisquare(df=2, size=(row_shape, column_shape))
            datasets = pd.DataFrame(chi_data)
        
        # Beta distribution
        if distribution == "Beta":
            beta_data = np.random.beta(a=2, b=2, size=(row_shape, column_shape))
            datasets = pd.DataFrame(beta_data)

        # Gamma distribution 
        if distribution == "Gamma":
            gamma_data = np.random.gamma(shape=2, scale=2, size=(row_shape, column_shape))
            datasets = pd.DataFrame(gamma_data)

        if text_data:
            # Generate text data
            vocab_size = int(row_shape*0.6)

            # Step 4.1: 生成词汇表
            vocab_dict = generate_vocabulary_with_llm(column_name, vocab_size=vocab_size)
            generated_table = self.generate_tabular_data(column_name, vocab_dict, row_shape)
            datasets = pd.DataFrame(generated_table, columns=[column_name])

        return datasets

    def generate_tabular_data_text(self, column_names, vocab_dict, row_shape, zipf_s=1.5, ngram_order=2, lda_topics=3):
        table = []

        # Step 2.1: 根据 Zipf's Law 分布生成单词
        def generate_zipf_vocab(vocab, s, size):
            word_probs = np.random.zipf(s, len(vocab))
            word_probs = word_probs / word_probs.sum()  # 归一化
            zipf_vocab = np.random.choice(vocab, size=size, p=word_probs)
            return zipf_vocab

        # Step 2.2: 生成 n-grams
        def generate_ngrams(vocab, n):
            ngrams = []
            for i in range(len(vocab) - n + 1):
                ngram = " ".join(vocab[i:i+n])
                ngrams.append(ngram)
            return ngrams

        # Step 2.3: LDA 主题分布
        def generate_lda_topics(vocab, n_topics):
            vectorizer = CountVectorizer(max_features=500)
            X_counts = vectorizer.fit_transform([" ".join(vocab)] * row_shape)
            lda = LatentDirichletAllocation(n_components=int(n_topics), random_state=42)
            lda.fit(X_counts)
            topic_distributions = lda.transform(X_counts)
            topics = np.argmax(topic_distributions, axis=1)
            return topics

        # Step 2.4: 生成 word embeddings
        def generate_word_embeddings(vocab):
            # 随机生成高维向量表示
            embedding_size = 50
            embeddings = {word: np.random.normal(0, 1, embedding_size) for word in vocab}
            return embeddings

        # Step 3: 生成表格数据
        for _ in range(row_shape):
            row = []
            for col in column_names:
                vocab = vocab_dict[col]  # 获取列对应的词汇表
                zipf_vocab = generate_zipf_vocab(vocab, zipf_s, len(vocab))
                ngram_vocab = generate_ngrams(zipf_vocab, ngram_order)
                topics = generate_lda_topics(zipf_vocab, lda_topics)
                word_embeddings = generate_word_embeddings(zipf_vocab)

                # 从生成的词汇中随机选择一个单词或短语
                word = random.choice(zipf_vocab)
                row.append(f"{word} (topic {random.choice(topics)})")
            table.append(row)

        return table


    def simulate_mask(self, data):
        """Generate missing value masks using different mechanisms"""
        mcar_mask = produce_NA(data, self.mcar_p, mecha="MCAR")
        mar_mask = produce_NA(data.values, self.mar_p, mecha="MAR")
        mnar_mask = produce_NA(data.values, self.mnar_p, mecha="MNAR")

        mask = mcar_mask & mar_mask & mnar_mask
        mask = mask.numpy().reshape(data.shape[0], data.shape[1])
        miss_data_x = data.copy()
        miss_data_x[mask == 0] = np.nan
        
        return pd.DataFrame(miss_data_x)
        
        