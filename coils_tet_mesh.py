import pandas as pd
import numpy as np
import cubit
from pymoab import core, types
import os
import sys


def extract_tabular_data(text):
    """
    Extract tabular data from the text of a .log file produced by Cubit.

    Arguments:
        text (str): The text content to extract data from.

    Returns:
        tuple: A tuple containing lists of function names, averages, standard deviations,
        minima, and maxima from the extracted data.
    """

    # Define the start and end markers for the tabular data in the text
    start_table = 'Function Name'
    end_table = 'Finished Command:'
    ranges = []
    start_index = 0

    # Loop to find all occurrences of the tabular data
    while True:
        # Find the start index of the table section
        start_index = text.find(start_table, start_index)
        if start_index == -1:
            break

        # Find the end index of the table section
        end_index = text.find(end_table, start_index + 1)
        if end_index == -1:
            break

        # Extract the text corresponding to the table section
        range_text = text[start_index: end_index + 1]
        ranges.append(range_text)

        # Move the start_index to the next position to find the next occurrence
        start_index = end_index + 1

    # Initialize lists to store the extracted data
    function_names = []
    averages = []
    std_devs = []
    minima = []
    maxima = []

    # Loop through each table section to extract relevant data
    for qual_metric in ranges:
        # Split the table section into rows using the newline character '\n'
        range_split = qual_metric[qual_metric.index('-\n'):].split(' ')

        # Extract relevant information from each row
        row_extraction = []
        for i in range_split:
            if i != '' and "(" not in i:
                row_extraction.append(i)

        # Calculate the offset to handle irregularities in the table structure
        offset = len(row_extraction) - 4
        row_extraction[0] = [''.join(row_extraction[:offset])]

        # Split and format the data into separate elements for each row
        for j in np.arange(1, 5):
            row_extraction[j] = row_extraction[offset + (j - 1): offset + j]
        row_extraction = row_extraction[:5]
        row_extraction[0][0] = row_extraction[0][0].split('-\n')[1]

        # Store the extracted data in the appropriate lists
        data_lists = [function_names, averages, std_devs, minima, maxima]
        indices = np.arange(5)
        for index, data_list in zip(indices, data_lists):
            if index == 0:
                data_list.append(row_extraction[index][0])
            else:
                data_list.append(float(row_extraction[index][0]))

    # Return the extracted data as a tuple
    return function_names, averages, std_devs, minima, maxima


def split_function_names(function_names):
    """
    Split the function names into multiple words based on uppercase letters.
        Necessary because the Cubit .log file makes the names only one word,
        but in order to match-case with the function names in Cubit commands,
        they need to be in the same format and thus split.

    Arguments:
        function_names (list): List of function names.

    Returns:
        list: List of function names with words separated.
    """

    # Initialize an empty list to store the split function names
    split_list = []

    # Loop through each function name in the input list
    for name in function_names:
        # Initialize an empty list to store the individual words of the function name
        split_words = []
        
        # Initialize an empty string to build the current word from characters
        current_word = ""

        # Loop through each character in the function name
        for char in name:
            # Check if the character is an uppercase letter
            if char.isupper() and current_word.isupper() and len(current_word) > 1:
                # If the character is uppercase and the current_word is also uppercase
                # and not a single character, consider it a separate word and append
                # the current_word to the split_words list.
                split_words.append(current_word)
                current_word = char
            elif char.isupper() and current_word and not current_word.isupper():
                # If the character is uppercase and the current_word exists but is not
                # uppercase, consider it a separate word and append the current_word
                # to the split_words list.
                split_words.append(current_word)
                current_word = char
            else:
                # If the character is not uppercase or doesn't meet the above conditions,
                # append it to the current_word to build the word.
                current_word += char

        # Append the last word to the split_words list
        split_words.append(current_word)
        
        # Join the split_words list to form the final split function name
        split_list.append(" ".join(split_words))
    
    # Update the function_names list with the split_list and return it
    function_names = split_list

    return function_names


def generate_coil_mesh(total_coil_count, log_general_path, cwd_option, general_export_path):
    """
    Create a tetrahedral mesh of magnet coils using Cubit and then apply
        quality metrics to the mesh. Store this data in a .log file.
    
        Arguments:
            total_coil_count (int): number of coils contained in the .step file
            log_general_path (str): absolute path to a .log file that will be
                created by Cubit. 
                **Note: Do not include any numerical index at the end of
                the file name because the function iterates over every coil in the .step
                file and will append the index to the .log filename.**
            cwd_option (bool): option to export all outputs to current working directory.
            general_export_path (str): absolute path (up to file name) of any export
                of the function, excluding file type.
        
        Returns:
            list: list of all of the absolute paths to the .log files generated.
    """

    # Define mesh quality metrics
    quality_metric_list = ['shape', 'aspect ratio', 'condition no.',
                           'distortion', 'element volume','equiangle skew', 'equivolume skew',
                           'inradius', 'jacobian', 'normalized inradius', 'relative size',
                           'scaled jacobian', 'shape and size']

    # Split the path for the log file
    log_split = log_general_path.split('.')
    log_prefix = log_split[0]
    log_ext = log_split[1]

    log_paths = []
    
    # For-loop iterates over all of the coils included in the coils.step file
    for coil_num in np.arange(1, int(total_coil_count) + 1):
        
         # Designate the file paths
        if cwd_option:
            log_path = f"{general_export_path}_{coil_num}.log"
        else:
            log_path = f"{log_prefix}_{coil_num}.{log_ext}"

        # Create the mesh
        cubit.cmd(f'volume {coil_num} scheme tetmesh')
        cubit.cmd(f'mesh volume {coil_num}')

        # Initiate logging
        cubit.cmd(f'logging on file "{log_path}"')

        # Run mesh quality metrics for each relevant metric
        for metric in quality_metric_list:
            qual_command = f'quality volume {coil_num} {metric} global draw mesh list'
            mesh_analytics = cubit.cmd(qual_command)

        log_paths.append(log_path)
    return log_paths


def log_to_dataframe(log_paths, total_coil_count):
    """
    Extract data from Cubit's .log file and store it in a Pandas Data Frame.
        Clean out implausible data as well, such as standard deviation of zero.

        Arguments:
            log_paths (list): list of all of the absolute paths to the
                .log files generated.
            total_coil_count (int): number of coils contained in the .step file.

        Returns:
            pandas.core.frame.DataFrame: Data Frame containing the data extracted
                and cleaned from the Cubit .log file.
                
    """

    # Create an empty Pandas data frame to store the top metric results
    top_results_cols = ['Function Name', 'Average','Standard Deviation',
                        'Minimum', 'Maximum']
    top_results_df = pd.DataFrame(columns = top_results_cols)

    for log in log_paths:
        # Read the log file
        with open(log, 'r') as file:
            text = file.read()

        # Extract the data from the log file using the extract_tabular_data function
        function_names, averages, std_devs, minima, maxima = extract_tabular_data(text)

        # Separate multi-word function names to be able to match-case with Cubit names
        function_names = split_function_names(function_names)

        # Compile the extracted tabular data into a Pandas data frame
        quality_metrics_df = pd.DataFrame({'Function Name': function_names,
                                            'Average': averages,
                                            'Standard Deviation': std_devs,
                                            'Minimum': minima,
                                            'Maximum': maxima})

        # Remove rows where the standard deviation equals 0
        # Found this for a few cases and it appeared too implausible to keep
        quality_metrics_df = quality_metrics_df.loc[quality_metrics_df['Standard Deviation'] != 0]

        # Sort rows by highest average quality metric and select the highest
        quality_metrics_df = quality_metrics_df.sort_values('Average', ascending=False)
        best_analytic = quality_metrics_df['Function Name'][0]

        # Save the best row into the top_results dataframe
        top_row_df = quality_metrics_df.iloc[0]
        top_row = list(top_row_df)
        row_df = pd.DataFrame([top_row], columns=top_results_cols)
        top_results_df = pd.concat([top_results_df, row_df])

    # Rename 'index' column to 'Coil Number'
    top_results_df.reset_index(inplace = True)
    top_results_df.rename(columns = {'index' : 'Coil Number'}, inplace = True)

    # Fix indexing for coil numbers
    top_results_df = top_results_df.transpose()
    top_results_df.iloc[0] = np.arange(1, int(total_coil_count) + 1)
    top_results_df = top_results_df.transpose()        

    return top_results_df


def coils_tet_mesh_export(cwd_option, general_export_path, exo_path, h5m_path):
    """
    Export tetrahedral meshes as EXODUS and .h5m files.

        Arguments:
            cwd_option (bool): option to export all outputs to current working directory.
            general_export_path (str): absolute path (up to file name) of any export
                of the function, excluding file type.
            exo_path (str): absolute path to a .exo file that will be
                created by the function.
            h5m_path (str): absolute path to a .h5m file that will be
                created by the function.
    """
    # Conditionally overwrites any paths assigned in function arguments
    if cwd_option:
        exo_path = f"{general_export_path}.exo"
        h5m_path = f"{general_export_path}.h5m"
    
    # EXODUS export
    cubit.cmd(f'export mesh "{exo_path}"')

    # Initialize the MOAB core instance
    mb = core.Core()

    # Load the EXODUS file
    exodus_set = mb.create_meshset()
    mb.load_file(exo_path, exodus_set)

    # Write the current coil's meshset to an individual .h5m file
    mb.write_file(h5m_path, [exodus_set])

def csv_exporter(top_results_df, cwd_option, general_export_path, csv_path):
    """
    Convert and export the Pandas Data Frame containing quality metrics data
        for the magnet coils into a .csv file.

        Arguments:
            top_results_df (pandas.core.frame.DataFrame): Data Frame containing the
                data extracted and cleaned from the Cubit .log file.
            cwd_option (bool): option to export all outputs to current working directory.
            general_export_path (str): absolute path (up to file name) of any export
                of the function, excluding file type.
            csv_path (str): absolute path to a .csv file that will store the top 
                quality metrics data for each coil. If cwd_option is True, then this 
                will be overwritten and can be set to an arbitrary string.            
    """
    # Conditionally overwrites any paths assigned in function arguments
    if cwd_option:
        csv_path = f"{general_export_path}.csv"

    #export the top_results data frame as a .csv
    top_results_df.to_csv(csv_path)
    

def coils_tet_mesh(coils_path, total_coil_count, cwd_option,
                    mesh_export_option, log_general_path, exo_path,
                    h5m_path, csv_export_option, csv_path):
    """
    Create and analyize tetrahedral mesh of the magnet coils using Cubit, with options
        for mesh export as EXODUS and .h5m files and data export as a .csv file.

        Arguments:
            coils_path (str): absolute path to the coils .step file
                containing all of the coil geometries to be meshed.
            total_coil_count (int): number of coils contained in the .step file.
            cwd_option (bool): option to export all outputs to current working directory.
            mesh_export_option (bool): option to export meshes as .exo and .h5m files.
            log_general_path (str): absolute path to a .log file that will be
                created by Cubit. 
                **Note: Do not include any numerical index at the end of
                the file name because the function iterates over every coil in the .step
                file and will append the index to the .log filename.**
            exo_path (str): absolute path to a .exo file that will be
                created by the function. If cwd_option is True, then this 
                will be overwritten and can be set to an arbitrary string.
            h5m_path (str): absolute path to a .h5m file that will be
                created by the function. If cwd_option is True, then this 
                will be overwritten and can be set to an arbitrary string.
            csv_option (bool): option to export a .csv file that will store
                the top quality metrics data for each coil.
            csv_path (str): absolute path to a .csv file that will store the top 
                quality metrics data for each coil. If cwd_option is True, then this 
                will be overwritten and can be set to an arbitrary string.
    """

    # Start Cubit
    cubit.init(['cubit', '-nojournal'])

    # Import coils
    cubit.cmd(f'import step "{coils_path}" heal')
    cubit.cmd('volume  size auto factor 5')

    # Get path to current working directory and define base file name
    cwd = os.getcwd()
    base_name = 'coils_tet_mesh'
    general_export_path = f"{cwd}/{base_name}"

    # Generate coil mesh and create quality metrics
    # Create log file as well as the output
    log_paths = generate_coil_mesh(total_coil_count, log_general_path, cwd_option, general_export_path)
    print("Meshing complete")

    # Extract data from the log into a Pandas Data Frame and clean it
    top_results_df = log_to_dataframe(log_paths, total_coil_count)
    print("Data loaded into Data Frame")

    # Conditionally export tetrahedral meshing
    if mesh_export_option:
        coils_tet_mesh_export(cwd_option, general_export_path, exo_path, h5m_path)
        print("Mesh exported")
    
    # Conditionally export the top_results_df data frame as a .csv
    if csv_export_option:
        csv_exporter(top_results_df, cwd_option, general_export_path, csv_path)
        print("CSV exported")

# Enable command-line execution
if __name__ == "__main__":
    arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 = sys.argv[1:10]
    coils_tet_mesh(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)