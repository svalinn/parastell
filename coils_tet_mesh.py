import pandas as pd
import numpy as np
import subprocess as sbp
import cubit


def coils_tet_mesh(coils_path, total_coil_count, log_general_path,
                   exo_general_path, h5m_general_path):
    """Create a tet mesh of the magnet coils using Cubit 
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
            exo_general_path (str): absolute path to a .exo file that will be
                created by the function. 
                **Note: Do not include any numerical index at the
                end of the file name because the function iterates over every coil in
                the .step file and will append the index to the .exo filename.**
            h5m_general_path (str): absolute path to a .h5m file that will be
                created by the function. 
                **Note: Do not include any numerical index at the
                end of the file name because the function iterates over every coil in
                the .step file and will append the index to the .h5m filename.**
        """

    # GEOMETRY AND MESH CREATION

    # Start Cubit - this step is key
    cubit.init(['cubit', '-nojournal'])

    # Import coils
    cubit.cmd(f'import step "{coils_path}" heal')
    cubit.cmd('volume  size auto factor 5')

    # For-loop iterates over all 48 coils included in the coils.step file
    for coil_num in np.arange(1,total_coil_count):
        
        # Create the mesh
        cubit.cmd(f'volume {coil_num} scheme tetmesh')
        cubit.cmd(f'volume {coil_num} scheme tetmesh')
        cubit.cmd(f'mesh volume {coil_num}')

        # Set log file path and initiate logging
        log_split = log_general_path.split('.')
        log_path = f"{log_split[0]}_{coil_num}.{log_split[1]})"
        cubit.cmd(f'logging on file "{log_path}"')

        # Mesh quality metrics
        quality_metric_list = ['shape', 'aspect ratio', 'condition no.', 'distortion', 'element volume',
                                'equiangle skew', 'equivolume skew','inradius', 'jacobian', 
                                'normalized inradius', 'relative size', 'scaled jacobian', 'shape and size']

        # Run mesh quality metrics for each relevant metric
        for metric in quality_metric_list:
            command = f'quality volume {coil_num} {metric} global draw mesh list'
            mesh_analytics = cubit.cmd(command)


        # MESH QUALITY ANALYSIS

        # Read the text file
        with open(log_path, 'r') as file:
            text = file.read()

        # Cut out each of the string 'tables' 
        start_table = 'Function Name'
        end_table = 'Finished Command:'
        ranges = []
        start_index = 0

        while True:
            # Find the starting index after the previous occurrence
            start_index = text.find(start_table, start_index)

            # Break the loop if the starting cue is not found
            if start_index == -1:
                break

            # Find the ending index after the starting index
            end_index = text.find(end_table, start_index + 1)

            # Break the loop if the ending cue is not found
            if end_index == -1:
                break

            # Extract the range between the starting and ending index
            range_text = text[start_index : end_index + 1]
            ranges.append(range_text)

            # Update the starting index for the next iteration
            start_index = end_index + 1

        # Extract the tabular data from the .log file and convert
        function_names = []
        averages = []
        std_devs = []
        minima = []
        maxima = []

        for qual_metric in ranges:
            range_split = qual_metric[qual_metric.index('-\n'):].split(' ')
            
            row_extraction = []
            for i in range_split:
                if i != '' and "(" not in i:
                    row_extraction.append(i)
            offset = len(row_extraction) - 4
            row_extraction[0] = [''.join(row_extraction[:offset])]
            
            for j in np.arange(1,5):
                row_extraction[j] = row_extraction[offset + (j-1) : offset + j]
            row_extraction = row_extraction[:5]
            row_extraction[0][0] = row_extraction[0][0].split('-\n')[1]

            data_lists = [function_names, averages, std_devs, minima, maxima]
            indices = np.arange(5)
            for index, data_list in zip(indices, data_lists):
                if index == 0:
                    data_list.append(row_extraction[index][0])
                else:
                    data_list.append(float(row_extraction[index][0]))

        # The above for-loop extracts the function names each as one word
        # In order to be able to later match-case with Cubit's libary, the following for-loop separates function names to multiple words if applicable
        split_list = []

        for name in function_names:
            split_words = []
            current_word = ""

            for char in name:
                if char.isupper() and current_word.isupper() and len(current_word) > 1:
                    split_words.append(current_word)
                    current_word = char
                elif char.isupper() and current_word and not current_word.isupper():
                    split_words.append(current_word)
                    current_word = char
                else:
                    current_word += char

            split_words.append(current_word)
            split_list.append(" ".join(split_words))
        function_names = split_list


        # Compile the .log data into a Pandas DataFrame
        quality_metrics_df = pd.DataFrame({'Function Name': function_names,
                                        'Average': averages,
                                        'Standard Deviation': std_devs,
                                        'Minimum': minima,
                                        'Maximum': maxima})

        # Clean the dataframe of implausible entries (e.g. found a standard dev of 0 -- seemed too unlikely)
        quality_metrics_df = quality_metrics_df.loc[quality_metrics_df['Standard Deviation'] != 0]

        # Sort data to find the highest average mesh quality
        quality_metrics_df = quality_metrics_df.sort_values('Average', ascending = False).reset_index()
        best_analytic = quality_metrics_df['Function Name'][0]
        print(best_analytic)


        # MESH EXPORT

        # Recreate the geometry and mesh with the best_analytic
        cubit.cmd('reset')
        cubit.cmd(f'import step "{coils_path}" heal')
        cubit.cmd('volume  size auto factor 5')
        cubit.cmd(f'volume {coil_num} scheme tetmesh')
        cubit.cmd(f'volume {coil_num} scheme tetmesh')
        cubit.cmd(f'mesh volume {coil_num}')
        cubit.cmd(f'mesh volume {coil_num}')
        cubit.cmd(f'quality volume {coil_num} {best_analytic} global draw mesh list')

        # Export the mesh as a .exo file
        exo_split = exo_general_path.split('.')
        exo_path = f"{exo_split[0]}_{coil_num}.{exo_split[1]}"
        cubit.cmd(f'export mesh "{exo_path}"')

        # Convert to h5m
        h5m_split = h5m_general_path.split('.')
        h5m_path = f"{h5m_split[0]}_{coil_num}.{h5m_split[1]}"
        h5m_conversion = sbp.run(f'mbconvert {exo_path} {h5m_path}', shell = True)