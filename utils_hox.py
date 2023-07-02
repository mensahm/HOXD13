# import modules

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import umap
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn import metrics
from scipy import stats



# define variables

pal_gli_list=['#de8f05', '#029e73','#0173b2','dimgrey']
pal_ins_list=['#de8f05', '#029e73','#0173b2']
pal_all_list=['#de8f05', '#029e73','#0173b2','#d55e00', '#cc78bc', '#ca9161']
pal_gli=sns.color_palette(pal_gli_list)
pal_ins=sns.color_palette(pal_ins_list)
pal_all=sns.color_palette(pal_all_list)

hue_order_mut=['ins7Ala', 'ins8Ala', 'ins9Ala', "missense", "truncating", "other"]
hue_order_gli=['ins7Ala', 'ins8Ala', 'ins9Ala', "GLI3"]
hue_order_ins=['ins7Ala', 'ins8Ala', 'ins9Ala']

features=["shape_mc3","shape_ph3","pos_mc3","pos_ph3","shape_mc1","shape_ph1","pos_mc1","pos_ph1","shape_mc5","shape_ph5","pos_mc5","pos_ph5","preax_adh","centr_adh","postax_adh","adh_mc","adh_prox_ph","adh_centr_ph","adh_dist_ph","adh_ph_5","extra_mc3","extra_pp3","extra_mp3","extra_dp3","extra_mc1","extra_pp1","extra_dp1","extra_mc5","extra_pp5","extra_mp5","extra_dp5","preax_rudiment","postax_rudiment","brachydactyly","clinodactyly","reference","ID"]




# define functions

def numval (e: str):
    """
    Converts a string (like specified in the radiographed table) into floats for each limb.
    
        Parameters:
            e (str): the string that should be converted to floats
    
        Returns:
            (rh,lh,rf,lf): a tuple with a numeric value for each limb
    """
    rh=0
    lh=0
    rf=0
    lf=0
    if "H" in e and "RH" not in e and "LH" not in e:
        e=e.replace("H", "RH, LH")
    if "F" in e and "RF" not in e and "LF" not in e:
        e=e.replace("F", "RF, LF")
    if "H" not in e and "F" not in e:
        e=e+"RH, LH, RF, LF"

    if "-" in e:
        if "RH" in e:
            rh=0
        if "LH" in e:
            lh=0
        if "RF" in e:
            rf=0
        if "LF" in e:
            lf=0
    elif "xx" in e or "xp" in e:
        if "RH" in e:
            rh=0.5
        if "LH" in e:
            lh=0.5
        if "RF" in e:
            rf=0.5
        if "LF" in e:
            lf=0.5
    elif "nx" in e or "nm" in e:
        if "RH" in e:
            rh=0.1
        if "LH" in e:
            lh=0.1
        if "RF" in e:
            rf=0.1
        if "LF" in e:
            lf=0.1
    elif "pr" in e:
        if "RH" in e:
            rh=0.8
        if "LH" in e:
            lh=0.8
        if "RF" in e:
            rf=0.8
        if "LF" in e:
            lf=0.8
    elif "2" in e:
        if "RH" in e:
            rh=1
        if "LH" in e:
            lh=1
        if "RF" in e:
            rf=1
        if "LF" in e:
            lf=1
    elif "3" in e:
        if "RH" in e:
            rh=2
        if "LH" in e:
            lh=2
        if "RF" in e:
            rf=2
        if "LF" in e:
            lf=2
    else:
        if "RH" in e:
            rh=1
        if "LH" in e:
            lh=1
        if "RF" in e:
            rf=1
        if "LF" in e:
            lf=1
            
    return (rh,lh,rf,lf)



def get_table (filepath: str, save_as: str, targets: list, features: list, len_to_right: int, function, separate_sides: bool = False, rows: tuple = (None,None)):
    """
    Converts the non-numeric table into a numeric table, which is saved as CSV.
                
            Parameters:
                filepath (str): data path of the table to be converted
                save_as (str): data path in which the generated table should be saved
                targets (list(str)): a list of the targets' column names
                features (list(str)): a list of the features' column names
                len_to_right (int): number of columns on the rigth side of the table that should not be converted
                function: the function to use for numerical transformatio
                separate_sides (bool): default=False, whether ro differentiate between left and right side
                rows (tuple): default=(None,None), subset of rows to be used
            
            Returns:
                newframe: a Pandas DataFrame of the numeric table          
    """
    
    headers=targets+features
    table=pd.read_csv(filepath, sep='\t', names=headers, index_col=False)
    table=table.drop(0, axis=0).reset_index(drop=True)
    table=table.iloc[rows[0]:rows[1],:]
    numtab=table[targets]
    for i in range (len(features)-len_to_right):
        column=headers[i+len(targets)]
        #print(f'column: {column}')
        if separate_sides is True:
            value_rh=[]
            value_lh=[]
            value_rf=[]
            value_lf=[]
        else:
            value_h=[]
            value_f=[]
        for n,x in enumerate (table[column]):
            print (n,x)
            content=x.split("/")
            numerics=[0,0,0,0]
            for y in content:
                numerics_to_add=function(y)
                numerics=[max(item1, item2) for item1, item2 in zip(numerics_to_add, numerics)]
            if separate_sides is True:
                value_rh+=[float(numerics[0])]
                value_lh+=[float(numerics[1])]
                value_rf+=[float(numerics[2])]
                value_lf+=[float(numerics[3])]
            else:
                value_h+=[float(max(numerics[0],numerics[1]))]
                value_f+=[float(max(numerics[2],numerics[3]))]
        index=numtab.shape[1]
        if separate_sides is True:
            numtab.insert(index,column+"_lf",value_lf)
            numtab.insert(index,column+"_rf",value_rf)
            numtab.insert(index,column+"_lh",value_lh)
            numtab.insert(index,column+"_rh",value_rh)
        else:
            numtab.insert(index,column+"_f",value_f)
            numtab.insert(index,column+"_h",value_h)
    newframe = numtab.copy()
    newframe.to_csv(save_as, index = False)
    return (newframe)




def load_table (data_name: str):
    """
    Loads CSV table as Pandas DataFrame.
    
        Parameters:
            data_name (str): data path of CSV table
            
        Returns:
            table as Pandas DataFrame
    """
    return(pd.read_csv(data_name,sep=",")) 


def standardize_data (df_train, n_targets: int, df_apply = None):
    """
    Standardizes numeric features using sklearn StandardScaler.
    
        Parameters:
            df_train: Pandas DataFrame to fit the StandardScaler onto
            n_targets (int): number of non-numeric columns of the DataFrame, beginning to count left
            df_apply: Pandas DataFrame to apply the StandardScaler onto; must have the same number of n_targets as df_train
        
        Returns:
            standardized_data (numpy.ndarray): data standardized by removing the mean and scaling to unit variance
    """
    headers=list(df_train.columns)
    features = headers[n_targets:]
    data_train = df_train.loc[:, features].values
    if df_apply is None:
        standardized_data = StandardScaler().fit_transform(data_train)
    else:
        data_apply = df_apply.loc[:, features].values
        standardized_data = StandardScaler().fit(data_train).transform(data_apply)
    return (standardized_data)


def get_headers (df):
    """
    To get a list of column names of a Pandas DataFrame.
    
        Parameters:
            df: Pandas DataFrame to get column names of
            
        Returns:
            headers: list of column names
    """
    headers=list(df.columns)
    return (headers)


def get_labels (df, target: str):
    """
    To get a list of the values of one column of a Pandas DataFrame.
    
        Parameters:
            df: Pandas DataFrame to get column values of
            target (str): name of the column of which the values should be taken
            
        Returns:
            headers: list of values
    """
    targets=list(df[target])
    return(targets)


def list_labels (df, target: str):
    """
    To get a list of the unique values of one column of a Pandas DataFrame.
    
        Parameters:
            df: Pandas DataFrame to get unique column values of
            target (str): name of the column of which the values should be taken
            
        Returns:
            headers: list of unique values
    """
    labels=list(df[target].unique())
    return(labels)


def get_pca (standardized_data_train, n_components: int, targets: list, target_values: list, standardized_data_apply=None):
    """
    Performs PCA with n principal components and appends given target columns.
    
        Parameters:
            standardized_data_train (numpy.ndarray): data standardized by removing the mean and scaling to unit variance (see standardize_data()) that shoul be used to compute the principal components
            n_components (int): number of desired principal components
            targets (list): list of column names for columns to be appended after PCA as possible labels
            target_values (list[list]): list of list(s) of values for columns to be appended after PCA as possible labels; the number of lists in the list should correspond to the length of the parameter targets (list)
            standardized_data_apply (numpy.ndarray): if default (= None), the same dataset that was used to fit the principal components is transformed; if another dataset is given in this parameter, this dataset will be transformed instead 
    
        Returns:
            princ_df: Pandas DataFrame containing the n principal components and target columns corresponding to the input in targets (list) and target_values (list[list])
    """
    col_names=[]
    for i in range(n_components):
        col_names+=[f"pr_component_{i+1}"]
    pca=PCA(n_components=n_components)
    princ_comp=pca.fit_transform(standardized_data_train)
    if standardized_data_apply is not None:
        princ_comp=pca.transform(standardized_data_apply)
    princ_df=pd.DataFrame(data=princ_comp, columns=col_names)
    for e,i in enumerate(targets):
             new_column=pd.DataFrame(data=target_values[e],columns=[i])
             princ_df=pd.concat([princ_df, new_column], axis = 1)
    print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
    return (princ_df)


def get_pca_weights (standardized_data, n_components: int):
    """
    Performs PCA with n principal components and returns weighting of the features for each principal component.
    
        Parameters:
            standardized_data (numpy.ndarray): data standardized by removing the mean and scaling to unit variance (see standardize_data())
            n (int): number of desired principal components
    
        Returns:
            weights (numpy.ndarray): weighting of the features for each principal component
    """
    pca=PCA(n_components=n_components)
    princ_comp=pca.fit_transform(standardized_data)
    weights = abs(pca.components_)
    return (weights)


def get_umap (standardized_data, df, n_neighbors=15, n_components=2, min_dist=0.1, random_state=None, targets: list = ['label'], standardized_data_apply=None):
    """
    Performs UMAP on standardized dataset and appends column "label" of original dataset.
        
        Parameters:
            standardized_data (numpy.ndarray): data standardized by removing the mean and scaling to unit variance (see standardize_data())
            df: original, not standardized dataset as Pandas DataFrame
            n_neighbors: default = 15, specifies the n_neighbors parameter of the UMAP object
            n_components: default = 2, specifies the n_components parameter of the UMAP object
            min_dist: default = 0.1, specifies the min_dist parameter of the UMAP object
            random_state: default = 1, specifies the random_state of the UMAP object
            targets (list): default = ['label'], list of columns of df, that should be concatenated with the final DataFrame
            standardized_data_apply (numpy.ndarray): if default (= None), the same dataset that was used to fit the principal components is transformed; if another dataset is given in this parameter, this dataset will be transformed instead 
        
        Returns:
            final_df_umap: Pandas DataFrame containing the n components as well as the column "label" from the original DataFrame
    """
    umap_object = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, random_state=random_state)
    umap_embedded = umap_object.fit_transform(standardized_data)
    if standardized_data_apply is not None:
        umap_embedded=umap_object.transform(standardized_data_apply)
    columns=[]
    for i in range (n_components):
        columns+=[f'component_{i+1}']
    umap_df=pd.DataFrame(data=umap_embedded, columns=columns)
    final_df_umap = pd.concat([umap_df, df[targets].reset_index(drop=True)], axis = 1)
    return (final_df_umap)




def get_tsne (standardized_data, df, lr="auto", perplexity=10, random_state=None, n_components=2):
    """
    Performs t-SNE on standardized dataset and appends column "label" of original dataset.
        
        Parameters:
            standardized_data (numpy.ndarray): data standardized by removing the mean and scaling to unit variance (see standardize_data()), used to fit the model and if standardized_data_apply=None, s
            df: original, not standardized dataset as Pandas DataFrame
            lr: default = "auto", specifies the learning rate parameter of the t-SNE object
            perplexity: default = 10, specifies the perplexity parameter of the t-SNE object
            random_state: default = None, specifies the random_state of the t-SNE object
            n_components: default = 2, specifies the n_components parameter of the t-SNE object
        
        Returns:
            final_df_tsne: Pandas DataFrame containing the n components as well as the column "label" from the original DataFrame
    """
    sne = TSNE(n_components=n_components,learning_rate=lr,perplexity=perplexity,init="random", random_state=random_state)
    tsne_embedded = sne.fit_transform(standardized_data)
    columns=[]
    for i in range (n_components):
        columns+=[f'component_{i+1}']
    tsne_df=pd.DataFrame(data=tsne_embedded, columns=columns)
    final_df_tsne = pd.concat([tsne_df, df[['label']]], axis = 1)
    return (final_df_tsne)




def plot (data, hue, filepath, hue_order=hue_order_ins, palette=hue_order_ins, with_kde=True, is_pca=False, with_annot=False, n_targets: int = 1, pca1=1, pca2=2):
    """
    Plots the first two dimensions of the given data, colored according to given hue.
        
        Parameters:
            data: Pandas DataFrame containing columns "component_1" and "component_2"
            hue (str): name of the column according to which the plot should be colored
            filepath (str): the filepath under which the figure should be saved; must end on .pdf or .png
            hue_order (list): default = None; a list defining the order of the different labels in the legend
            palette: default = pal_all; a colorpalette that should be used to color the plot
            with_kde (bool): default=True, whether to plot a seaborn kdeplot underneath the data points
            is_pca (bool): default=False, whether the data is pca data
            with_annot (bool): default=False; wether to annotate the patient ID to each datapoint; this requires a column "ID" in pca_data
            pca1 (int): default = 1; first principal component to be plotted
            pca2 (int): default = 2; second principal component to be plotted
            n_targets (int): deefault=1; number of columns on the right end of data that are non-numerical and should not be considered for plotting
            
        no return 
    """
    if is_pca is True:
        xcomp, ycomp, xlabel, ylabel = (f"pr_component_{pca1}", f"pr_component_{pca2}", f'Principal Component {pca1}', f'Principal Component {pca2}')
    else:
        xcomp, ycomp, xlabel, ylabel = ("component_1", "component_2", 'Component 1', 'Component 2')
    plt.figure(figsize=(8,8))
    sns.set_context('paper', font_scale=1.4)
    sns.set_style("white")
    if with_kde is True:
        sns.kdeplot(x=xcomp, y=ycomp, data=data, hue=hue, palette=palette, hue_order=hue_order, fill=True, alpha=0.6, levels=4, thresh=0.4)
    p=sns.scatterplot(x=xcomp, y=ycomp, data=data, hue=hue, s=70, palette=palette, hue_order=hue_order)
    if with_annot is True:
        for line in data.index.values.tolist():
            plt.text(data.component_1[line]+0.1, data.component_2[line], data.ID[line], horizontalalignment='left', size='small', color='black', weight='semibold')
    plt.xlabel(xlabel, fontsize=20, labelpad=15)
    plt.ylabel(ylabel, fontsize=20)
    plt.savefig(filepath)



    
def kde_plots(filepath: str, with_gli3: bool, name:str, rounded: bool):
    """
    Creates kernel density estimate plots using each PCA, t-SNE and UMAP to reduce the given data.
    
    Parameters:
        filepath (str): the filepath of the table to be loaded
        with_gli3 (bool): wether the data contains GLI3 patients or not
        name (str): a name for the table that is used as part of the filename
        rounded (bool): whether to use no values between 0 and 1
    
    no returns

    """
    df=load_table(filepath)
    if rounded is True:
        df=df.replace({0.1:0., 0.2:0., 0.5:1., 0.8:1.})
    if with_gli3 is False:
        rand_tsne, rand_umap = 9,14
        hue_order=hue_order_ins
        palette=pal_ins
    else:
        rand_tsne, rand_umap = 0,22
        hue_order=hue_order_gli
        palette=pal_gli
    standardized_data=standardize_data(df,2)
    pca_data=get_pca(standardized_data, 4, ["label"], [get_labels(df, "label")])
    plot(pca_data,"label",f"kde_pca12_{name}.pdf",hue_order=hue_order,palette=palette, is_pca=True)
    plot(pca_data,"label",f"kde_pca24_{name}.pdf",hue_order=hue_order,palette=palette, is_pca=True, pca1=2, pca2=4)
    
    def rotate(angle):
        ax.view_init(azim=angle)

    # The following code is copied from: http://blog.mahler83.net/2019/10/rotating-3d-t-sne-animated-gif-scatterplot-with-matplotlib/
    hue_dict={'ins7Ala':'#de8f05', 'ins8Ala':'#029e73', 'ins9Ala':'#0173b2', 'GLI3':'dimgrey'}
    hues=[hue_dict[i] for i in pca_data["label"]]
    fig=plt.figure(figsize=(8,8))
    ax=plt.axes(projection="3d")
    if with_gli3 is True:
        ax.scatter(pca_data["pr_component_1"], pca_data["pr_component_2"], pca_data["pr_component_3"], c=hues, s=100)
    else:
        ax.scatter(pca_data["pr_component_1"], pca_data["pr_component_2"], pca_data["pr_component_3"], c=hues, s=100)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(f'3d_pca_{name}123.gif', writer=animation.PillowWriter(fps=20))
    # End of copied code
    
    tsne_data=get_tsne(standardized_data, df, lr="auto", perplexity=5, random_state=rand_tsne)
    plot(tsne_data, "label", f"kde_tsne_{name}.pdf", hue_order=hue_order,palette=palette)
    umap_data=get_umap(standardized_data, df, n_neighbors=10, min_dist=0.0, n_components=2, random_state=rand_umap)
    plot(umap_data, "label", f"kde_umap_{name}.pdf", hue_order=hue_order,palette=palette)
    
    
    
def decision_tree (data, labels: list, label_names: list, depth: int, file_name: str, with_gli3: bool = False):
    """
    Trains a decision tree on the given data, saves it as PNG file and prints out the accuracy of the decision tree on the training dataset.
        Parameters:
            data: Pandas DataFrame of the data to fit the decision tree on
            labels (list): list of values for class membership of the datapoints, in the same order as in data
            label_names (list): list of unique classes
            depth (int): specifies the maximum depth of the decision tree
            file_name (str): filepath and name, where the PNG file should be stored; must end with .png
            with_gli3 (bool): whether the dataset contains GLI3 cases or not
            
        Returns:
            dec_tree: the fitted decision tree object
    """
    feature_cols=list(data.columns)
    dec_tree=DecisionTreeClassifier(max_depth=depth)
    tree_data=dec_tree.fit(data, labels)
    dot_data=StringIO()
    export_graphviz(dec_tree, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feature_cols,class_names=label_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    
    if with_gli3 is False:
        colors = ['#de8f05', '#029e73','#0173b2']
    else:
        colors = ['dimgrey', '#de8f05', '#029e73','#0173b2']
    nodes = graph.get_node_list()
    for node in nodes:
        if node.get_label():
            values = [int(ii) for ii in node.get_label().split('value = [')[1].split(']')[0].split(',')]
            values = [int(255 * v / sum(values)) for v in values]
            index_color = np.argmax(values)
            node.set_fillcolor(colors[index_color])
            
    graph.write_png(file_name)
    Image(graph.create_png())
    predicted=dec_tree.predict(data)
    print("Accuracy:",metrics.accuracy_score(labels, predicted))
    return(dec_tree)

    

        
def calculate_decision_trees(depths: list, tree_tables: list):
    """
    Generates decision trees for all depths in the list depths and all datasets in the list tree_tables with and without performing PCA beforehands.
    
        Parameters:
            depths (list(int)): list of integers with maximum depths to be used
            tree_tables (list): list containing a filepath, an alphabetically ordered list of class names, a name for the table that is used as part of the filename, the number of non-numeric columns on the left of the given table and a boolean, whether the dataset includes GLI3 cases or not; in the given order
        
        no returns
    """
    for depth in depths:
        for element in tree_tables:
            table=load_table(element[0]).replace({0.1:0.0, 0.2:0.0, 0.5:0.0, 0.8:1.0})
            labels=get_labels(table, "label")
            data=table.iloc[:,element[3]:]
            print (f'{element[2]}Tree{depth}')
            decision_tree(data,labels, element[1], depth, f'{element[2]}Tree{depth}.png', with_gli3=element[4])
            stand=pd.DataFrame(standardize_data(data, 0))
            data=get_pca(stand, 20, ["label"], [labels]).iloc[:,:-1]
            print (f'{element[2]}PcaTree{depth}')
            decision_tree(data,labels, element[1], depth, f'{element[2]}PcaTree{depth}.png', with_gli3=element[4])
            

            
def severity_scores (filepath: str):
    """
    Computes the severity score derived from the severity score published by Guo et al. (2021) per individual.
    
    Parameters:
        filepath (str): the filepath of the table to be loaded; values must be encoded as not present", "present", "probable", "not assessable" or "limb externally normal"
    
    no returns
    """
    table=load_table(filepath).replace({"not present":0.0, "present":1.0, "probable":0.8, "not assessable":0.5, "limb externally normal":0.2})
    scores=[]
    for i in range(len(table)):
        score=0
        for j, symptom in enumerate(["Webbed fingers/toes", "Brachydactyly", "Position of phalanges", "Polydactyly"]):
            for limb in ["RH", "LH", "RF", "LF"]:
                """if j==3:
                    score += round(max(table.iloc[i][f"{symptom[0]}_{limb}"],table.iloc[i][f"{symptom[1]}_{limb}"],table.iloc[i][f"{symptom[2]}_{limb}"],table.iloc[i][f"{symptom[3]}_{limb}"],table.iloc[i][f"{symptom[4]}_{limb}"]))
                else:"""
                score += round(table.iloc[i][f"{symptom} {limb}"])
        scores+=[score]
    table["severity_score"]=scores
    return table



def severity_scores_perfam (filepath: str):
    """
    Computes the severity score derived from the severity score published by Guo et al. (2021) per family.
    
    Parameters:
        filepath (str): the filepath of the table to be loaded; values must be encoded as not present", "present", "probable", "not assessable" or "limb externally normal"
    
    no returns
    """
    table=load_table(filepath).replace({"not present":0.0, "present":1.0, "probable":0.8, "not assessable":0.5, "limb externally normal":0.2})
    scores=[]
    for i in range(len(table)):
        score=0
        for j, symptom in enumerate(["Webbed fingers/toes", "Brachydactyly", "Position of phalanges", "Polydactyly"]):
            for limb in ["RH", "LH", "RF", "LF"]:
                """if j==3:
                    score += round(max(table.iloc[i][f"{symptom[0]}_{limb}"],table.iloc[i][f"{symptom[1]}_{limb}"],table.iloc[i][f"{symptom[2]}_{limb}"],table.iloc[i][f"{symptom[3]}_{limb}"],table.iloc[i][f"{symptom[4]}_{limb}"]))
                else:"""
                score += round(table.iloc[i][f"{symptom} {limb}"])
        scores+=[score]
    table["severity_score"]=scores
    means=pd.DataFrame(table[["Label","Family","severity_score"]])
    means=means.groupby(['Family','Label']).mean().reset_index()
    return means



def boxplot_severity (data, filename: str):
    """
    Generates and saves a boxplot of the given severity scores stratified by mutation type.
    
    Parameters:
        data: Pandas DataFrame containing the columns "label" and "severity_score"
        filename (str): a file name for the plot
    
    no returns
    """
    scores = [i for i in data['severity_score']]
    labels = [i for i in data['Label']]
    ins7Ala_scores = [score for s, score in enumerate(scores) if labels[s] == 'ins7Ala']
    ins8Ala_scores = [score for s, score in enumerate(scores) if labels[s] == 'ins8Ala']
    ins9Ala_scores = [score for s, score in enumerate(scores) if labels[s] == 'ins9Ala']
    truncating_scores = [score for s, score in enumerate(scores) if labels[s] == 'truncating']
    missense_scores = [score for s, score in enumerate(scores) if labels[s] == 'missense']
    other_scores = [score for s, score in enumerate(scores) if labels[s] == 'other']
    all_scores = [ins7Ala_scores, ins8Ala_scores, ins9Ala_scores, truncating_scores, missense_scores, other_scores]
    all_labels = ['ins7Ala', 'ins8Ala', 'ins9Ala', 'truncating', 'missense', 'other']    
    
    fig=plt.figure(figsize=(24,8))
    sns.boxplot(data=data, x="Label", y="severity_score", order=["ins7Ala", "ins8Ala", "ins9Ala", "truncating", "missense", "other"], width= 0.6, palette=pal_all)
    plt.xlabel("Variant Type", fontsize=25, labelpad=20)
    plt.ylabel("Severity Score", fontsize=25, labelpad=5)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    comparisons = []
    for i in range(len(all_scores)):
        for j in range(len(all_scores)-i-1):
            comparisons.append([i, i+j+1])
    ls=[]
    for n, i in enumerate(comparisons):
        t, p = (stats.mannwhitneyu(all_scores[i[0]], all_scores[i[1]]))
        level =  'n.s.'
        if p <0.05:
            level = '*'
        if p <0.01:
            level = '**'
        if p <0.001:
            level = '***'
        
        x = i[1]-i[0]
        l= 10
        if x == 1: l = 0
        if x == 2: l = 1 + i[0]%2
        if x == 3: l = 3 + i[0]%3
        if x == 4: l = 6 + i[0]%4
        if x == 5: l = 8 + i[0]%5
        print(all_labels[i[0]]+' vs. '+all_labels[i[1]]+': p =', p, level, l)
        if level != 'n.s.' and l not in ls:
            ls.append(l)
        
    ls.sort()    
    
    for n, i in enumerate(comparisons):
        t, p = (stats.mannwhitneyu(all_scores[i[0]], all_scores[i[1]]))
        level =  'n.s.'
        if p <0.05:
            level = '*'
        if p <0.01:
            level = '**'
        if p <0.001:
            level = '***'
        
        x = i[1]-i[0]
        l= 10
        if x == 1: l = 0
        if x == 2: l = 1 + i[0]%2
        if x == 3: l = 3 + i[0]%3
        if x == 4: l = 6 + i[0]%4
        if x == 5: l = 8 + i[0]%5
        if level != 'n.s.':
            l = ls.index(l)
            y, h, col = 14.5, 0.25, 'black'
            y = y+1.2*l
            plt.plot([i[0]+0.15, i[0]+0.15, i[1]-0.15, i[1]-0.15], [y, y+h, y+h, y], lw=1.5, c=col)
            plt.text((i[0]+i[1])*.5, y+h, ' p = ' + str("%.3g" % p), ha='center', va='bottom', color=col, fontsize = 'xx-large')
    
    plt.savefig(f"{filename}.pdf")
    
    
def plot_heatmap (filepath: str, n_targets: int):
    """
    Loads a given table, performs PCA on it and creates a CSV-table containing the weighting of the first two principal components. This can be used for further work up to manually create a heatmap as shown in Fig. 3.
    
    Parameters:
        filepath (str): the filepath of the table to be loaded
        n_targets (int): number of non-numeric columns of the DataFrame, beginning to count left
    
    no returns
    """
    num_tab=load_table(filepath)
    stand_tab=standardize_data(num_tab, n_targets)
    headers_for_weights=get_headers(num_tab)[n_targets:]
    df_weights=pd.DataFrame(data=get_pca_weights(standardize_data(num_tab, n_targets), 2), columns=headers_for_weights, index=["Principal Component 1", "Principal Component 2"])
    df_weights=df_weights.transpose().round(decimals=2)
    df_weights.to_csv("weightings.csv")
    
    



