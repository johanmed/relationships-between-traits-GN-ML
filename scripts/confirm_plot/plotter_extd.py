#!/usr/bin/env python3
"""
Script 23

This script plots GWAS Manhattan of associated/correlated traits for visual confirmation

It can also mimic QTL plots by drawing peaks on GWAS plots, provided the trait is quantitative

Inputs:
- selected data: ../../../diabetes_gemma_association_data_plrt_filtered_traits_selected.csv


Adapted from plotter. Source code can be found at https://github.com/matchcase/plotter/blob/master/plot.py

"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# To suppress the palette warning
import warnings
warnings.filterwarnings("ignore")

debug_flag = False


def draw_manhattan_plot(df, draw_peak, threshold_value, hovering_enabled, bin_size):

    df.chr = df.chr.astype('category')
    category_order = df.chr.unique()
    df.chr = df.chr.cat.set_categories(category_order, ordered=True)
    shiftpos = 0
    
    cpos = []
    for chromosome, group_df in df.groupby('chr', observed=False):
        cpos.append(group_df['pos'] + shiftpos)
        shiftpos += group_df['pos'].max()
        
    df['cpos'] = pd.concat(cpos)
    
    sns.set_theme()
    sns.set_style(rc = {'axes.facecolor': "#eeeeee", 'grid.color': "#f0f0f0"})
    palette_col = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#56b4e9', '#949494']
    
    manhattan_plot = sns.relplot(
        data=df,
        alpha=0.7,
        x='pos',
        y='-logP',
        hue='full_desc',
        palette=manhattan_palette,
        linewidth=0,
        legend=None
       )
       
    manhattan_plot.ax.set_ylim(-0.05, None)
    manhattan_plot.ax.set_ylabel('-log P', rotation=0, labelpad=24)
    cpos_spacing = (df.groupby('chr', observed=False)['cpos'].max()).iloc[0]
    cpos_spacing = cpos_spacing - (df.groupby('chr', observed=False)['cpos'].min()).iloc[0]
    cpos_spacing = cpos_spacing/20
    manhattan_plot.ax.set_xlim(df['cpos'].min() - cpos_spacing, df['cpos'].max() + cpos_spacing)
    
    if len(df["chr"].unique()) > 1:
        manhattan_plot.ax.set_xlabel('Chromosome')
        manhattan_plot.ax.set_xticks(df.groupby('chr', observed=False)['cpos'].median())
        manhattan_plot.ax.xaxis.grid(False)
        manhattan_plot.ax.set_xticklabels(df['chr'].unique())
    else:
        manhattan_plot.ax.set_xlabel('position')
        xtick_step = len(df['pos']) // 10
        manhattan_plot.ax.set_xticks(df['pos'][::xtick_step])
        
    prev_tick = 0.0
    span_color = 'lightgrey'
    for idx, tick in enumerate(df.groupby('chr', observed=False)['cpos'].min()):
        if debug_flag:
            print("Enumerating:", idx, tick)
        if idx != 'N/A':
            manhattan_plot.ax.axvspan(prev_tick, tick, facecolor=span_color, zorder=0, alpha=0.5)
        prev_tick = tick
        span_color = '#ccccee' if span_color == 'lightgrey' else 'lightgrey'
    last_tick = (df.groupby('chr', observed=False)['cpos'].max()).iloc[-1]
    manhattan_plot.ax.axvspan(prev_tick, last_tick, facecolor=span_color, zorder=0, alpha=0.5)
    
    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.95, right=0.9)
    
    if draw_peak:
        maxlp = df.loc[df['-logP'].idxmax()]
        manhattan_plot.ax.axvline(x=maxlp['cpos'],
                                  color=sns.color_palette('deep')[3],
                                  linestyle='dashed',
                                  linewidth=1)
    if threshold_value:
        manhattan_plot.ax.axhline(y=threshold_value,
                                  color=sns.color_palette('deep')[3],
                                  linestyle='dashed',
                                  linewidth=1)
   
    plt.show()






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process filtered dataset concerning specific selection of traits')
    
    parser.add_argument('file', type=str, help='Path to the file to process')
    
    parser.add_argument('--peak',
                        help='Draw a vertical line through the peak value',
                        action='store_true')
                        
    parser.add_argument('--threshold',
                        type=float,
                        help='Draw a threshold line at a given -logP value')
                        
    parser.add_argument('--hover',
                        help='Show details of the point that the cursor is hovering on',
                        action='store_true')
                        
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debugging', default=False)
    
    parser.add_argument('--trait', default=None, nargs='?', type=str, help='Trait name for AraQTL file')
    
    parser.add_argument('--chromosome', default=None, nargs='?', type=str, help='Selected chromosome')
    
    parser.add_argument('--bin-size', default=100000, nargs='?', type=int, help='Bin size for SNP density')


    args = parser.parse_args()
    _, file_extension = os.path.splitext(args.file)
    if file_extension.lower() == '.csv':
        data = parse_csv_file(args.file)
    else:
        print(f"Unsupported file extension: {file_extension}. Please provide a CSV file.")
        exit(1)
    debug_flag = args.debug
    
    if args.chromosome:
        data = data[data["chr"] == args.chromosome]
        if debug_flag:
            print("Succeeded in picking chromosome!")
            
    
    
    draw_manhattan_plot(data, args.peak, args.threshold, args.hover, args.bin_size)
