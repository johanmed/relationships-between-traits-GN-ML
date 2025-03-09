#!/usr/bin/env python3
"""
Script 23

This script GWAS or QTL plots of associated/correlated traits for visual confirmation

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
import textalloc as ta
from sklearn.neighbors import NearestNeighbors

# To suppress the palette warning
import warnings
warnings.filterwarnings("ignore")

debug_flag = False


def draw_manhattan_plot(df, draw_peak, threshold_value, hovering_enabled):
    
    # Define cpos using to chr and pos sorting
    
    df.chr = df.chr.astype('category')
    category_order = df.chr.unique()
    df.chr = df.chr.cat.set_categories(category_order, ordered=True)
    shiftpos = 0
    
    cpos = []
    for chromosome, group_df in df.groupby('chr', observed=False):
        cpos.append(group_df['pos'] + shiftpos)
        shiftpos += group_df['pos'].max()
        
    df['cpos'] = pd.concat(cpos)
    
    # Define layout
    
    sns.set_theme()
    sns.set_style(rc = {'axes.facecolor': "#eeeeee", 'grid.color': "#f0f0f0"})
    palette_col = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#56b4e9', '#949494']
    
    # Plot relational plot
    
    manhattan_plot = sns.relplot(
        data=df,
        alpha=0.7,
        x='cpos',
        y='-logP',
        hue='full_desc',
        palette=palette_col,
        linewidth=0,
        legend='auto'
       )
    
    # Make extra layout configurations
    
    manhattan_plot.figure.set_size_inches(20, 20)
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
    
    # Draw peak line if asked
    if draw_peak:
        maxlp = df.loc[df['-logP'].idxmax()]
        manhattan_plot.ax.axvline(x=maxlp['cpos'],
                                  color=sns.color_palette('deep')[3],
                                  linestyle='dashed',
                                  linewidth=1)
    # Draw threshold line if asked
    if threshold_value:
        manhattan_plot.ax.axhline(y=threshold_value,
                                  color=sns.color_palette('deep')[3],
                                  linestyle='dashed',
                                  linewidth=1)
                                  
    
    
    text_list = []
    x_list = []
    y_list = []
    skip_lines = (not not threshold_value) + (not not draw_peak)
    def clear_points_and_lines():
        for line_idx, line in enumerate(plt.gca().lines):
            if line_idx < skip_lines:
                if debug_flag:
                    print("Skipped a line")
                continue
            line.remove()
        for text in manhattan_plot.ax.texts:
            if text != hover_annot:
                text.remove()
    
    
    def on_click(event):
        if event.button == 3 and hovering_enabled:
            create_or_destroy_hover_annot()
            manhattan_plot.fig.canvas.draw_idle()
            return
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            
            if debug_flag:
                print(x, y)
                
            data = df[['cpos', '-logP']].values
            point = np.array([x, y])
            
            neigh=NearestNeighbors(n_neighbors=1, algorithm='auto')
            neigh.fit(data)
            
            dist, ind = neigh.kneighbors(point)
            
            closest_point_index = ind[0][0]
            
            marker_attribute = df.loc[closest_point_index, 'marker']
            x_attribute = df.loc[closest_point_index, 'cpos']
            y_attribute = df.loc[closest_point_index, '-logP']
            if debug_flag:
                print(x_attribute, y_attribute)
                print(f"Clicked on point with marker attribute: {marker_attribute}")
            if marker_attribute not in text_list:
                if debug_flag:
                    print("New attribute!")
                text_list.append(marker_attribute)
                x_list.append(x_attribute)
                y_list.append(y_attribute)
            else:
                if debug_flag:
                    print("Existing attribute: deleting!")
                idx = text_list.index(marker_attribute)
                deleted_flag = False
                for text_obj in manhattan_plot.ax.texts:
                    if text_obj.get_text() == marker_attribute:
                        text_obj.remove()
                        if debug_flag:
                            print("Found the text object!")
                        deleted_flag = True
                        break
                if not deleted_flag:
                    print("ERROR: Deleted Flag not satisfied for point!")
                else:
                    deleted_flag = False
                text_list.pop(idx)
                x_list.pop(idx)
                y_list.pop(idx)
            clear_points_and_lines()
            ta.allocate_text(fig=manhattan_plot.figure,
                                         ax=manhattan_plot.ax,
                                         x=x_list,
                                         y=y_list,
                                         text_list=text_list,
                                         linecolor=[sns.color_palette('deep')[3]]*len(x_list),
                                         textsize=12)
            plt.draw()
    hover_annot = manhattan_plot.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                             textcoords="offset points",
                                             bbox=dict(boxstyle="round",
                                                       fc=(0.94, 0.95, 0.9)),
                                             arrowprops=dict(arrowstyle="->",
                                                             color="b"))
    if not hovering_enabled:
        hover_annot.set_visible(False)
    def create_or_destroy_hover_annot():
        nonlocal hover_annot
        if hover_annot.get_visible():
            hover_annot.set_visible(False)
        else:
            hover_annot.set_visible(True)
    def update_hover_annot(event):
        x, y = event.xdata, event.ydata
        if debug_flag:
            print(x, y)
        closest_point_index = (((df['cpos'] - x)/df['cpos'])**2 + ((df['-logP'] - y)/df['-logP'])**2).idxmin()
        marker_attribute = df.loc[closest_point_index, 'marker']
        x_attribute = df.loc[closest_point_index, 'cpos']
        y_attribute = df.loc[closest_point_index, '-logP']
        txt = "Pos: "+ str(x_attribute) + ", -logP: " + str(y_attribute) + ": "+ marker_attribute
        hover_annot.set_text(txt)
        hover_annot.xy = (x_attribute, y_attribute)
        hover_annot.get_bbox_patch().set_alpha(0.4)

    def on_hover(event):
        if not hover_annot.get_visible():
            return
        update_hover_annot(event)
        manhattan_plot.fig.canvas.draw_idle()

    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    if hovering_enabled:
        plt.gcf().canvas.mpl_connect('motion_notify_event', on_hover)
    
    
    
    
    plt.legend(loc='upper right')
    manhattan_plot._legend.remove()
    
    manhattan_plot.figure.suptitle('Overlapping GWAS plots for selection of traits', fontsize=20)
    
    plt.show()
    
    #plt.savefig('../../output/Overlapping_GWAS_plots_selection_traits', dpi=500)







def draw_qtl_plot(df, draw_peak, threshold_value, hovering_enabled):
    
    # Define cpos using to chr and pos sorting
    
    df.chr = df.chr.astype('category')
    category_order = df.chr.unique()
    df.chr = df.chr.cat.set_categories(category_order, ordered=True)
    shiftpos = 0
    
    cpos = []
    for chromosome, group_df in df.groupby('chr', observed=False):
        cpos.append(group_df['pos'] + shiftpos)
        shiftpos += group_df['pos'].max()
        
    df['cpos'] = pd.concat(cpos)
    
    # Define layout
    
    sns.set_theme()
    sns.set_style(rc = {'axes.facecolor': "#eeeeee", 'grid.color': "#f0f0f0"})
    palette_col = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#56b4e9', '#949494']
    
    # Plot relational plot
    
    qtl_plot = sns.relplot(
        data=df,
        alpha=0.7,
        x='cpos',
        y='-logP',
        hue='full_desc',
        palette=palette_col,
        legend='auto', 
        kind='line',
        linewidth=1
       )
    
    # Make extra layout configurations
    
    qtl_plot.figure.set_size_inches(20, 20)
    
    for line in qtl_plot.ax.lines:
        x, y = line.get_xydata().T
        qtl_plot.ax.fill_between(x, -0.05, y, color=line.get_color(), alpha=0.3)
        
    qtl_plot.ax.set_ylim(-0.05, None)
    qtl_plot.ax.set_ylabel('-log P', rotation=0, labelpad=24)
    cpos_spacing = (df.groupby('chr', observed=False)['cpos'].max()).iloc[0]
    cpos_spacing = cpos_spacing - (df.groupby('chr', observed=False)['cpos'].min()).iloc[0]
    cpos_spacing = cpos_spacing/20
    qtl_plot.ax.set_xlim(df['cpos'].min() - cpos_spacing, df['cpos'].max() + cpos_spacing)
    
    if len(df["chr"].unique()) > 1:
        qtl_plot.ax.set_xlabel('Chromosome')
        qtl_plot.ax.set_xticks(df.groupby('chr', observed=False)['cpos'].median())
        qtl_plot.ax.xaxis.grid(False)
        qtl_plot.ax.set_xticklabels(df['chr'].unique())
    else:
        qtl_plot.ax.set_xlabel('position')
        xtick_step = len(df['pos']) // 10
        qtl_plot.ax.set_xticks(df['pos'][::xtick_step])
        
    prev_tick = 0.0
    span_color = 'lightgrey'
    for idx, tick in enumerate(df.groupby('chr', observed=False)['cpos'].min()):
        if debug_flag:
            print("Enumerating:", idx, tick)
        if idx != 'N/A':
            qtl_plot.ax.axvspan(prev_tick, tick, facecolor=span_color, zorder=0, alpha=0.5)
        prev_tick = tick
        span_color = '#ccccee' if span_color == 'lightgrey' else 'lightgrey'
    last_tick = (df.groupby('chr', observed=False)['cpos'].max()).iloc[-1]
    qtl_plot.ax.axvspan(prev_tick, last_tick, facecolor=span_color, zorder=0, alpha=0.5)
    
    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.95, right=0.9)
    
    # Draw peak line if asked
    if draw_peak:
        maxlp = df.loc[df['-logP'].idxmax()]
        qtl_plot.ax.axvline(x=maxlp['cpos'],
                                  color=sns.color_palette('deep')[3],
                                  linestyle='dashed',
                                  linewidth=1)
    # Draw threshold line if asked
    if threshold_value:
        qtl_plot.ax.axhline(y=threshold_value,
                                  color=sns.color_palette('deep')[3],
                                  linestyle='dashed',
                                  linewidth=1)
    
    
    
    text_list = []
    x_list = []
    y_list = []
    # As this is a line plot, we need to skip one line each for each chromosome!
    skip_lines = (not not threshold_value) + (not not draw_peak) + len(df.chr.unique())
    if debug_flag:
        print("Skipping", skip_lines, "lines.")
    def clear_points_and_lines():
        for line_idx, line in enumerate(plt.gca().lines):
            if line_idx < skip_lines:
                if debug_flag:
                    print("Skipped a line")
                continue
            line.remove()
        for text in qtl_plot.ax.texts:
            if text != hover_annot:
                text.remove()
    
    
    
    def on_click(event):
        if event.button == 3 and hovering_enabled:
            create_or_destroy_hover_annot()
            qtl_plot.fig.canvas.draw_idle()
            return
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            if debug_flag:
                print(x, y)
            closest_point_index = (((df['cpos'] - x)/df['cpos'])**2 + ((df['-logP'] - y)/df['-logP'])**2).idxmin()
            marker_attribute = df.loc[closest_point_index, 'marker']
            x_attribute = df.loc[closest_point_index, 'cpos']
            y_attribute = df.loc[closest_point_index, '-logP']
            if debug_flag:
                print(x_attribute, y_attribute)
                print(f"Clicked on point with marker attribute: {marker_attribute}")
            if marker_attribute not in text_list:
                if debug_flag:
                    print("New attribute!")
                text_list.append(marker_attribute)
                x_list.append(x_attribute)
                y_list.append(y_attribute)
            else:
                if debug_flag:
                    print("Existing attribute: deleting!")
                idx = text_list.index(marker_attribute)
                deleted_flag = False
                for text_obj in qtl_plot.ax.texts:
                    if text_obj.get_text() == marker_attribute:
                        text_obj.remove()
                        if debug_flag:
                            print("Found the text object!")
                        deleted_flag = True
                        break
                if not deleted_flag:
                    print("ERROR: Deleted Flag not satisfied for point!")
                else:
                    deleted_flag = False
                text_list.pop(idx)
                x_list.pop(idx)
                y_list.pop(idx)
            clear_points_and_lines()
            ta.allocate_text(fig=qtl_plot.figure,
                                         ax=qtl_plot.ax,
                                         x=x_list,
                                         y=y_list,
                                         text_list=text_list,
                                         linecolor=[sns.color_palette('deep')[3]]*len(x_list),
                                         textsize=12)
            plt.draw()
    hover_annot = qtl_plot.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                             textcoords="offset points",
                                             bbox=dict(boxstyle="round",
                                                       fc=(0.94, 0.95, 0.9)),
                                             arrowprops=dict(arrowstyle="->",
                                                             color="b"))
    if not hovering_enabled:
        hover_annot.set_visible(False)
    def create_or_destroy_hover_annot():
        nonlocal hover_annot
        if hover_annot.get_visible():
            hover_annot.set_visible(False)
        else:
            hover_annot.set_visible(True)
    def update_hover_annot(event):
        x, y = event.xdata, event.ydata
        if debug_flag:
            print(x, y)
        closest_point_index = (((df['cpos'] - x)/df['cpos'])**2 + ((df['LOD'] - y)/df['LOD'])**2).idxmin()
        marker_attribute = df.loc[closest_point_index, 'marker']
        x_attribute = df.loc[closest_point_index, 'cpos']
        y_attribute = df.loc[closest_point_index, 'LOD']
        txt = "Pos: "+ str(x_attribute) + ", LOD: " + str(y_attribute) + ": " + marker_attribute
        hover_annot.set_text(txt)
        hover_annot.xy = (x_attribute, y_attribute)
        hover_annot.get_bbox_patch().set_alpha(0.4)

    def on_hover(event):
        if not hover_annot.get_visible():
            return
        update_hover_annot(event)
        qtl_plot.fig.canvas.draw_idle()

    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    if hovering_enabled:
        plt.gcf().canvas.mpl_connect('motion_notify_event', on_hover)
    
    
    
    
    
    plt.legend(loc='upper right')
    qtl_plot._legend.remove()
    
    qtl_plot.figure.suptitle('Overlapping QTL plots for selection of traits', fontsize=20)
    
    plt.show()
    
    #plt.savefig('../../output/Overlapping_QTL_plots_selection_traits', dpi=500)
    
    
    

def parse_csv_file(file):
    # Manage parsing of csv file
    df = pd.read_csv(file, sep=',', header=0)
    return df



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process filtered dataset concerning specific selection of traits')
    
    parser.add_argument('file', type=str, help='Path to the file to process')
    
    parser.add_argument('--peak',
                        help='Draw a vertical line through the peak value',
                        action='store_true')
                        
    parser.add_argument('--threshold',
                        type=float,
                        help='Draw a threshold line at a given -logP value')
                        
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debugging', default=False)
    
    parser.add_argument('--type', type=str, default='gwas', help='Specify the type of plot to draw')
    
    parser.add_argument('--hover', help='Show details of the point that the cursor is hovering on', action='store_true')
    

    args = parser.parse_args()
    
    _, file_extension = os.path.splitext(args.file)
    
    if file_extension.lower() == '.csv':
        data = parse_csv_file(args.file)
    else:
        print(f"Unsupported file extension: {file_extension}. Please provide a CSV file.")
        exit(1)
        
    debug_flag = args.debug
    
    
    # Proceed to drawing
    if args.type=='gwas':
        draw_manhattan_plot(data, args.peak, args.threshold, args.hover) 
        
    elif args.type=='qtl':
        draw_qtl_plot(data, args.peak, args.threshold, args.hover)
