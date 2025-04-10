#!/usr/bin/env python3

"""
Script 28

This script takes the list of edges (pairs of trait and gene)

It builds an undirected network



"""

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import lay_render

import matplotlib.pyplot as plt
import csv

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build genetic network for specific selection of traits')
    
    parser.add_argument('file', type=str, help='Path to the file containing traits and genes of interest')
    
    args = parser.parse_args()

    with open(args.file) as tg_file:
    
        csv_reader = csv.reader(tg_file) # read pairs of traits and genes
    
        graph = nx.Graph(csv_reader) # create undirected network
        
        
        traits = [input('Please enter the first trait:'), input('Please enter the second trait:'), input('Please enter the third trait:'), input('Please enter the fourth trait:')]
    
    
        # Label nodes as traits or genes for distinction

        traits_dict = {node : (node in traits) for node in graph} # return boolean telling if node is a trait or not
    
        nx.set_node_attributes(graph, traits_dict, 'trait') # set attribute trait for each node
    
    
        # Save network as graphml for flexibility of sharing and improvement
    
        nx.write_graphml(graph, f'../../output/network_selection_traits.graphml')
    
    
        # Proceed to layout
    
        colors = ['seashell' if node[1]['trait'] else 'lightsteelblue' for node in graph.nodes(data=True)]
    
        lay_render.attrs['node_color'] = colors
    
        _, plot = plt.subplots(figsize=(20, 20))
    
        pos = graphviz_layout(graph)
    
        # Proceed to rendering
    
        nx.draw_networkx(graph, pos, **lay_render.attrs)
    
        plot.set_title(f'Genetic network for selection of traits', fontsize=20)
        
        plot.set_axis_off()
    
        lay_render.set_extent(pos, plot)
    
        lay_render.plot(f'network_selection_traits', save=True) # save and plot network for visualization
    
    
