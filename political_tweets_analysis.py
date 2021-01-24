from community import community_louvain
import networkx as nx
import os
import json
import pandas as pd
from datetime import datetime


class Helpers:



    def is_weighted(self,net):
    """
    @Name : is_weighted
    @Do: check if a given network contain weighted edges.
    @ Param:
            net - graph - a networkx object
    @ Return:
            True if weighted else False
    """

        weighted = False
        for u,v in net.edges():
            if net.get_edge_data(u,v) != {}:
                weighted = True
                break
        return weighted


    def cluster(self,partition):
    '''
    @Name : cluster
    @Do: create data structure contains for each community the list of it's members
    @ Param:
            partition - a dictionary with pair of key:value
                        where key = node name & value = community identifier
                        (which community each node belong to)
    @ Return:
            result_dict - a dictionary with pair of key:value
                        where key = community identifier  & value = list of nodes belong to this community
    '''

        result_dict = {}
        for key in partition.keys():
            if partition[key] in result_dict.keys():
                result_dict[partition[key]].append(key)
            else:
                result_dict[partition[key]] = []
                result_dict[partition[key]].append(key)
        return result_dict

    '''
    @Name : get Lc
    @Do: find the number of links within a given community
    @ Param:
            edges_list - list of all edges in a given network
            community_list - list od all members in the community
    @ Return:
            count - number of links between community members
    '''

    def get_Lc(self,edges_list,community_list):
        count = 0
        for link in edges_list:
            if link[0] in community_list and link[1] in community_list:
                count += 1
        return count

    '''
    @Name : get Kc
    @Do: find the sum degree number for all nodes in the community
    @ Param:
            network - a networkx object contain nodes and edges.
            community_list - list od all members in the community
    @ Return:
            count - the sum degree of all nodes
    '''

    def get_Kc(self,network,community_lst):
        count = 0
        new_dict = {}
        for key,value in network.degree():
            new_dict[key] = value
        for par in community_lst:
            if par in new_dict.keys():
                count += new_dict[par]
        return count

    '''
    @Name : modularity
    @Do: find the modularity of a given network
    @ Param:
            network - a networkx object contain nodes and edges.
            partition - a given partition of certain network
    @ Return:
            modularity - number in [0,1] where 1 is a complete community.
    '''


    def modularity(self,netwotk,partition):
        modularity = 0
        # partition = self.remove_dups(partition)
        # partition = self.add_singeltons(netwotk,partition)
        for community in partition.values():
            l_c = self.get_Lc(netwotk.edges(),community)
            k_c = self.get_Kc(netwotk,community)
            l = len(netwotk.edges())
            m_c = (l_c/l) - ((k_c/(2*l))**2)
            modularity += m_c
        return modularity


    '''
    @Name : turn_to_par_dic
    @Do: create data structure contains for each community the list of it's members
    @ Param:
            gen - a generator object contain the output from the girvin-newman algorithm.
    @ Return:
            par_dic - a dictionary contains for each community, list of it's members.
    '''

    def turn_to_par_dic(self,gen):
        temp_partition = [i for i in next(gen)]
        par_dic = {}
        for i in range(0, len(temp_partition)):
            for j in temp_partition[i]:
                par_dic[j] = i
        return par_dic

    def unpack_gen(self,gen):
        lst = [list(i) for i in gen]
        new_dic = {}
        for i in range(0,len(lst)):
            new_dic[i] = []
        for j in range(0, len(lst)):
            for obj in lst[j]:
                new_dic[j].append(obj)
        return new_dic

'''
@Name: community_detector
@Do: for a given network and a given algorithm, calculate the modularity, partitions and number of communities.
@ Param:
        algorithm_name (str) -
            Name of the algorithm to run (girvin_newman, louvain or clique_percolation)
            most_valualble_edge(None or function) -
                A parameter that is used only by the ‘girvin_newman’ algorithm.
                networkX object.

@ Return:
        ‘num_partitions’ - Number of partitions the network was divided to
        ‘modularity’ - The modularity value of the partition
        ‘partition’ -
            The partition of the network.
            Each element in the list is a community detected (with node names).
'''
def community_detector(algorithm_name,network,most_valualble_edge=None):
    if Helpers().is_weighted(network):
        weight = 'weight'
    else:
        weight = None
    if algorithm_name == 'louvain':
        par_var = community_louvain.best_partition(network,weight=weight)
        num_partitions = len(set(par_var.values()))
        modularity = community_louvain.modularity(par_var,network)
        cliques_dict = {}
        for i in range(0,num_partitions):
            cliques_dict[i] = []
        for key,value in par_var.items():
            cliques_dict[value].append(key)

        partition = [lst for lst in cliques_dict.values()]

        return {'num_partitions': num_partitions,
                      'modularity' : modularity,
                      'partition' : partition}

    elif algorithm_name == 'girvin_newman':
        gen_par = nx.algorithms.community.centrality.girvan_newman(network,most_valualble_edge)
        mod = 0
        num_partitions = 0
        partition = None
        i=0
        list_of_losers = []
        while True:
            try:
                if len(list_of_losers) > 100:
                    break
                i+=1
                dic = {}
                temp_partition = next(gen_par)
                for index, community in enumerate(temp_partition):
                    dic[index] = list(community)
                temp_mod = Helpers().modularity(network, dic)
                if temp_mod > mod:
                    mod = temp_mod
                    num_partitions = len(dic)
                    partition = [v for v in dic.values()]
                    list_of_losers = []
                else:
                    list_of_losers.append(temp_mod)
            except StopIteration:
                break
        return {'num_partitions': num_partitions,
               'modularity': mod,
               'partition': partition}

    elif algorithm_name == 'clique_percolation':
        flag = True
        modularity = 0
        num_p = 0
        par = None
        for num in range(2,len(network.nodes())):
            try:
                if flag == False:
                    break
                clique_per = nx.algorithms.community.k_clique_communities(network,num)
                clique_per = Helpers().unpack_gen(clique_per)
                temp_mod = Helpers().modularity(network,clique_per.copy())
                if temp_mod > modularity:
                    modularity = temp_mod
                    num_p = len([i for i in clique_per.values()])
                    par = [val for val in clique_per.values()]
            except IndexError:
                flag = False
        return {'num_partitions': num_p,
               'modularity': modularity,
               'partition': par}





'''
@Name: edge_selector_optimizer
@Do: in order to increase the efficiency of the girvan-newman algorithm,
        this function will try to find the best match in each step of
        the algorithm the best edge to pull out from the original
        network and

@ Param:
        network - graph - networkx object
@ Return:
        ‘num_partitions’ - Number of partitions the network was divided to
        ‘modularity’ - The modularity value of the partition
        ‘partition’ -
            The partition of the network.
            Each element in the list is a community detected (with node names).
'''

def edge_selector_optimizer(network):
    if Helpers().is_weighted(network):
        weights_list = [(e,list(network.get_edge_data(*e).values())[0]) for e in network.edges()]
        weights_list = sorted(weights_list,key = lambda x:x[1],reverse = True)
        return weights_list[0][0]
    else:
        betweenness = nx.edge_betweenness_centrality(network)
        lst_of_betweenness = [(k,v) for k,v in betweenness.items()]
        return max(lst_of_betweenness,key=lambda x:x[1])[0]



'''
This class contain help function.
'''

class Assistant:
    '''
    @Name: get_df
    @Do: read all txt files in a given path, and convert them from json foramt to pandas dataframe.
    @ Param:
            file_path - string. location of all files
            start_date - a date from
    @ Return:

    '''

    def get_df(self, file_path, start, end):
        start = datetime.strptime(start, '%Y-%m-%d')
        end = datetime.strptime(end, '%Y-%m-%d')

        dates_lst = pd.date_range(start='2019-01-15', end='2019-04-15')
        dates_lst = [(dates_lst[i], i) for i in range(0, len(dates_lst))]
        all_tweets = pd.DataFrame()
        i_start = 0
        i_end = 0
        for date in dates_lst:
            if date[0] == start:
                i_start = date[1]
            elif date[0] == end:
                i_end = date[1]
            else:
                continue

        files_list = [file for file in os.listdir(file_path) if file.endswith('.txt')]

        for i, file in enumerate(files_list):
            if i < i_start:
                continue
            elif i > i_end:
                break
            else:
                with open(file_path + '/' + file) as f:
                    try:
                        all_tweets = all_tweets.append([json.loads(line) for line in f], ignore_index=True)
                    except json.JSONDecodeError:
                        pass
        return all_tweets.dropna(subset=['retweeted_status'])

    def get_dic(self, df):
        dic = {}
        count = 0
        for i in range(0, df.shape[0]):
            try:
                id = df.iloc[i]['user']['id']
                tweet_id = df.iloc[i]['retweeted_status']['user']['id']
                key = (id, tweet_id)
                if key in dic:
                    dic[key] += 1
                else:
                    dic[key] = 1
            except KeyError:
                count += 1
        return dic

    def is_political(self, edge, pol_lst):
        if (edge[0] in pol_lst) and (edge[1] in pol_lst):
            return True
        return False

    def deal_with_pol(self, n, final_dic, pol_list):
        nodes = set(set([k[0] for k in final_dic.keys()]).union([k[1] for k in final_dic.keys()]))
        if n > len(nodes):
            return final_dic
        f_dic = final_dic.copy()
        sort_list = sorted([(k, v) for k, v in final_dic.items()], key=lambda x: x[1], reverse=True)
        not_pol = [i for i in sort_list if not self.is_political(i[0], pol_list)]
        i = 0
        j = 0
        keys_to_keep = []
        temp_dic = {}
        while i < n:
            cur_node = sort_list[j][0][0]
            if cur_node in temp_dic:
                j += 1
            else:
                keys_to_keep += [i for i in sort_list if ((i[0][0] == cur_node) and (i[0][1] in pol_list))]
                temp_dic[cur_node] = None
                i += 1

        only_keys = [k[0] for k in keys_to_keep]
        for key in list(final_dic.keys()):
            if (self.is_political(key, pol_list)) or (key in only_keys):
                continue
            else:
                del f_dic[key]

        return f_dic


'''
Param:
    @files_path - 
        Location of all files the function requires.
        These are the 90 txt files and the central_political_players.csv file
    @start_date - First day to include (format: YYYY-MM-DD)
    @end_date - Last day to include (format: YYYY-MM-DD)
    @non_parliamentarians_nodes - number of edges that are not politicians allowed to insert the dictionary

'''


def construct_heb_edges(files_path, start_date='2019-03-15', end_date='2019-04-15', non_parliamentarians_nodes=0):
    assistant = Assistant()
    tweets_df = assistant.get_df(files_path, start_date, end_date)
    final_dic = assistant.get_dic(tweets_df)
    pol_path = files_path + '/' + 'central_political_players.csv'
    politicians = pd.read_csv(pol_path)
    pol_id_list = list(politicians['id'])
    final_dic = assistant.deal_with_pol(non_parliamentarians_nodes, final_dic, pol_id_list)
    return final_dic


def construct_heb_network(edge_dictionary):
    G = nx.Graph()
    for key in edge_dictionary.keys():
        G.add_edge(key[0], key[1], weight=edge_dictionary[key])
    return G




