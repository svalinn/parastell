import pandas as pd
import numpy as np
import cubit
from pymoab import core, types


def extract_tabular_data(text):
    """
    Extract tabular data from the text of a .log file produced by Cubit.

    Args:
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

    Args:
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


def coils_tet_mesh(coils_path, total_coil_count, log_general_path,
                   exo_path, h5m_path, csv_path):
    """Create a tetrahedral mesh of the magnet coils using Cubit 
    and export them as .exo and .h5m files

        Arguments:
            coils_path (str): absolute path to the coils .step file
                containing all of the coil geometries to be meshed.
            total_coil_count (int): number of coils contained in the .step file
            log_general_path (str): absolute path to a .log file that will be
                created by Cubit. 
                **Note: Do not include any numerical index at the end of
                the file name because the function iterates over every coil in the .step
                file and will append the index to the .log filename.**
            exo_path (str): absolute path to a .exo file that will be
                created by the function. 
            h5m_path (str): absolute path to a .h5m file that will be
                created by the function. 
            csv_path (str): absolute path to a .csv file that will store the top 
                quality metrics data for each coil.
        """

    # GEOMETRY AND MESH CREATION

    # Start Cubit - this step is key
    cubit.init(['cubit', '-nojournal'])

    # Import coils
    cubit.cmd(f'import step "{coils_path}" heal')
    cubit.cmd('volume  size auto factor 5')

    # Split log path
    log_split = log_general_path.split('.')
    log_prefix = log_split[0]
    log_ext = log_split[1]

    # Define mesh quality metrics
    quality_metric_list = ['shape', 'aspect ratio', 'condition no.',
                           'distortion', 'element volume','equiangle skew', 'equivolume skew',
                           'inradius', 'jacobian', 'normalized inradius', 'relative size',
                           'scaled jacobian', 'shape and size']
    
    # Create an empty Pandas data frame to store the top metric results
    top_results_cols = ['Function Name', 'Average','Standard Deviation',
                                        'Minimum', 'Maximum']
    top_results = pd.DataFrame(columns = top_results_cols)

    # For-loop iterates over all of the coils included in the coils.step file
    for coil_num in np.arange(1,total_coil_count + 1):
        
        # Create the mesh
        cubit.cmd(f'volume {coil_num} scheme tetmesh')
        cubit.cmd(f'volume {coil_num} scheme tetmesh')
        cubit.cmd(f'mesh volume {coil_num}')

        # Set log file path and initiate logging
        log_path = f"{log_prefix}_{coil_num}.{log_ext}"
        cubit.cmd(f'logging on file "{log_path}"')

        # Run mesh quality metrics for each relevant metric
        for metric in quality_metric_list:
            qual_command = f'quality volume {coil_num} {metric} global draw mesh list'
            mesh_analytics = cubit.cmd(qual_command)


        # MESH QUALITY ANALYSIS

        # Read the log file
        with open(log_path, 'r') as file:
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

        # Override the current mesh analytic with the best quality metric
        cubit.cmd(f'quality volume {coil_num} {best_analytic} global draw mesh list')

        # Save the best row into the top_results dataframe
        top_row_df = quality_metrics_df.iloc[0]
        top_row = list(top_row_df)
        row_df = pd.DataFrame([top_row], columns=top_results_cols)
        top_results = pd.concat([top_results, row_df])
 

    # MESH EXPORT

    # Export the mesh as an EXODUS file
    cubit.cmd(f'export mesh "{exo_path}"')

    # Initialize the MOAB core instance
    mb = core.Core()

    # Load the EXODUS file
    exodus_set = mb.create_meshset()
    mb.load_file(exo_path, exodus_set)
    
    # Create a new meshset for each coil iteration
    coil_meshset = mb.create_meshset()

    # Add the current coil's mesh to the meshset
    mb.add_entities(coil_meshset, [exodus_set])

    # Write the current coil's meshset to an individual .h5m file
    mb.write_file(h5m_path, [coil_meshset])
    

    # CSV EXPORT

    # Rename 'index' column to 'Coil Number'
    top_results.reset_index(inplace = True)
    top_results.rename(columns = {'index' : 'Coil Number'}, inplace = True)

    # Fix indexing for coil numbers
    top_results = top_results.transpose()
    top_results.iloc[0] = np.arange(1, total_coil_count + 1)
    top_results = top_results.transpose()
    print(top_results.head())

    # Export the top_results data frame as a .csv
    top_results.to_csv(csv_path)
