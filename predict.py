################
# Import Libraries
################
from io import StringIO

import seaborn as sns
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
# import warnings

# do this to make Pandas show all the columns of a DataFrame, otherwise it just shows a summary
pd.set_option('display.max_columns', None)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Poverty Prediction")
st.sidebar.title("Analysing Option")
st.title('Poverty Prediction')

img = Image.open("head.jpeg")
st.image(img, width=600)
uploaded_file = st.sidebar.file_uploader("Choose a File")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(bytes_data)

    string_data = stringio.read()
    df_train = pd.read_csv(uploaded_file)
    train_id = df_train['Id']

    train_idhogar = df_train['idhogar']

    df_train.drop(columns=['Id'], inplace=True)
    if st.sidebar.checkbox("Training Data"):
        st.header("""
            Training Data
            ***
        """)
        st.write(df_train.describe())
    st.write("Shape of train data: ", df_train.shape)

    all_data = df_train

    st.write("A glimpse at the columns of training data:")
    st.write(df_train.head())
    st.markdown("### These are the core data fields as described in the data description:")
    st.markdown("- *Id* - a unique identifier for each row.")
    st.markdown("- *Target* - the target is an ordinal variable indicating groups of income levels. 1 = extreme "
                "poverty 2 = moderate poverty 3 = vulnerable households 4 = non vulnerable households")
    st.markdown("- idhogar - this is a unique identifier for each household. This can be used to create "
                "household-wide features, etc. All rows in a given household will have a matching value for this "
                "identifier.")
    st.markdown("- *parentesco1* - indicates if this person is the head of the household")
    st.write("""Let's see description of Target:""")

    st.write(df_train['Target'].describe())

    st.markdown(" ### We need to make predictions on a household level whereas we have been given the data at an "
                "individual level.")
    st.markdown("That is why during all our analysis we will focus only on those columns which have parentesco1 == 1.")
    st.markdown("These are columns for the heads of households and each household has only one head.")


    def barplot_with_anotate(feature_list, y_values, plotting_space=plt, annotate_vals=None):
        x_pos = np.arange(len(feature_list))
        plotting_space.bar(x_pos, y_values)
        plotting_space.xticks(x_pos, feature_list, rotation=270)
        if annotate_vals is None:
            annotate_vals = y_values
        for i in range(len(feature_list)):
            plotting_space.text(x=x_pos[i] - 0.3, y=y_values[i] + 1.0, s=annotate_vals[i])


    df_train_heads = df_train.loc[df_train['parentesco1'] == 1]
    poverty_label_sizes = list(df_train_heads.groupby('Target').size())

    barplot_with_anotate(['extreme', 'moderate', 'vulnerable', 'non-vulnerable'], poverty_label_sizes,
                         annotate_vals=[str(round((count / df_train_heads.shape[0]) * 100, 2)) + '%'
                                        for count in poverty_label_sizes])
    plt.rcParams["figure.figsize"] = [6, 6];
    plt.xlabel('Poverty Label')
    plt.ylabel('No. of People')
    st.pyplot()

    st.markdown("So, we can see that **a majority( > 65 %) of the households fall within the Non - vulnerable "
                "category**.This means that we are dealing with an imbalanced classification problem.")

    st.subheader("Home Life for various poverty groups")


    def plot_dwelling_property(property_df):
        _, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(16, 16))

        target_idx = 0
        for row in range(2):
            for col in range(2):
                percentage_list = [round((count / poverty_label_sizes[target_idx]) * 100, 2)
                                   for count in list(property_df.iloc[target_idx, :])]
                x_pos = list(range(len(property_df.columns)))

                axarr[row, col].bar(x_pos,
                                    percentage_list,
                                    color='y')

                axarr[row, col].set_title('For individuals in Poverty group=' + str(target_idx + 1))

                xtick_labels = list(property_df.columns)
                xtick_labels.insert(0, '')  # insert a blank coz 'set_xticklabels()' skips the ist element

                axarr[row, col].set_xticklabels(xtick_labels, rotation=300)

                axarr[row, col].set_ylim(bottom=0, top=100)

                for i in range(len(property_df.columns)):
                    axarr[row, col].annotate(xy=(x_pos[i] - 0.3, percentage_list[i] + 1.0), s=percentage_list[i])

                axarr[0, 0].set_ylabel("Percentage of the total in this poverty group")
                axarr[1, 0].set_ylabel("Percentage of the total in this poverty group")
                axarr[1, 0].set_xlabel("Types")
                axarr[1, 1].set_xlabel("Types")

                axarr[row, col].autoscale(enable=True, axis='x')
                target_idx += 1


    st.write("### **Outside wall material of the house:**")

    outside_wall_material_df = df_train_heads.groupby('Target').sum()[['paredblolad', 'paredzocalo',
                                                                       'paredpreb', 'pareddes', 'paredmad', 'paredzinc',
                                                                       'paredfibras', 'paredother']]
    outside_wall_material_df
    st.markdown("**paredblolad =1** if predominant material on the outside wall is block or brick")
    st.markdown("**paredzocalo =1** if predominant material on the outside wall is socket (wood, zinc or absbesto")
    st.markdown("**paredpreb =1** if predominant material on the outside wall is prefabricated or cement")
    st.markdown("**pareddes	=1** if predominant material on the outside wall is waste material")
    st.markdown("**paredmad	=1** if predominant material on the outside wall is wood ")
    st.markdown("**paredzinc =1** if predominant material on the outside wall is zink")
    st.markdown("**paredfibras =1** if predominant material on the outside wall is natural fibers")
    st.markdown("**paredother =1** if predominant material on the outside wall is other")
    st.pyplot(plot_dwelling_property(outside_wall_material_df))

    st.markdown("We see that a majority(69.24 %) of the households under poverty group 4(non - vulnerable) have brick "
                "wall on the outside.")
    st.markdown("- As we go from there to group 1(extreme), the percentage of houses having brick wall "
                "decreases.Cement wall and wood walls become increasingly more common.")
    st.markdown("- The top 3 most common types of wall material across all the groups are( in descending order of "
                "popularity) - brick > prefabricated or cement > wood")

    st.write("""
    ***
    """)
    st.write("### Floor material of the house")

    floor_material_df = df_train_heads.groupby('Target').sum()[
        ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']]
    floor_material_df

    st.markdown("**pisomoscer =1** if predominant material on the floor is mosaic, ceramic, terrazo")
    st.markdown("**pisocemento =1** if predominant material on the floor is cement")
    st.markdown("**pisoother =1** if predominant material on the floor is other")
    st.markdown("**pisonatur =1** if predominant material on the floor is  natural material")
    st.markdown("**pisonotiene =1** if no floor at the household")
    st.markdown("**pisomadera =1** if predominant material on the floor is wood")

    st.pyplot(plot_dwelling_property(floor_material_df))

    st.markdown("- We see that a majority of households belonging to poverty group 3(vulnerable) and group 4(non - "
                "vulnerable) have mossaic, ceramic, tazzo floors(62.82 % and 79.38 % respectively")
    st.markdown("- This floor type becomes less common as we move across from group 4 to group 1 and other types("
                "especially the cemented floors) become more common.")
    st.markdown("- The top 3 most common types of floors across all the groups are( in descending order of "
                "popularity) - mossaic, ceramic, tazzo > cemented > wooden")

    st.write("""
        ***
        """)
    st.write("## **Toilet:**")

    toilet_df = df_train_heads.groupby('Target').sum()[
        ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']]
    toilet_df

    st.pyplot(plot_dwelling_property(toilet_df))

    st.markdown("- A large majority of the households have a toilet connected to a septic tank(73 % - 81 %).")
    st.markdown("- A toilet connected to sewer or cess pool becomes more common as we move to group 4. It is probably "
                "better, more expensive type of installation.")
    st.write("""
           ***
           """)
    st.write("## **Rubbish Disposal**")

    rubbish_disposal_df = df_train_heads.groupby('Target').sum()[
        ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']]
    rubbish_disposal_df

    st.pyplot(plot_dwelling_property(rubbish_disposal_df))

    st.write("""
           ***
           """)
    st.write("### **Roof material of the house:**")

    roof_material_df = df_train_heads.groupby('Target').sum()[['techozinc', 'techoentrepiso', 'techocane', 'techootro']]
    roof_material_df

    st.pyplot(plot_dwelling_property(roof_material_df))

    st.write("""
           ***
           """)
    st.write("### **Water Provision**")

    water_provision_df = df_train_heads.groupby('Target').sum()[['abastaguadentro', 'abastaguafuera', 'abastaguano']]
    water_provision_df

    st.pyplot(plot_dwelling_property(water_provision_df))

    st.write("""
           ***
           """)
    st.write("### **Electricity**")

    electricity_df = df_train_heads.groupby('Target').sum()[['public', 'planpri', 'noelec', 'coopele']]
    electricity_df

    st.pyplot(plot_dwelling_property(electricity_df))

    st.write("""
           ***
           """)

    st.write("### **Main source of energy in cooking**")

    cooking_energy_df = df_train_heads.groupby('Target').sum()[
        ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']]
    cooking_energy_df

    st.pyplot(plot_dwelling_property(cooking_energy_df))

    st.markdown("- Here, we see that gas and electricity are the major sources of energy in the kitchens for all the "
                "people.")
    st.markdown("- For the poverty group 4 (non-vulnerable), electricity is slightly more popular than gas whereas in "
                "the other groups, gas is the more popular choice.")
    st.markdown("- As we move from group 1 (extreme) to group 4 (non-vulnerable), the popularity of electricity "
                "increases and that of gas decreases.")
    st.write("""
           ***
           """)
    st.write("### **Household Size**")

    avg_household_size_df = df_train_heads.groupby('Target').mean()['hhsize']
    avg_household_size_df

    st.write(df_train.groupby('Target').mean().head())

    st.write("""
           ***
           """)
    st.write("### **Urban or rural**")

    urban_rural_df = df_train_heads.groupby('Target').sum()[['area1', 'area2']]
    urban_rural_df['UrbanPercentage'] = urban_rural_df['area1'] * round((100 / sum(urban_rural_df['area1'])), 6)
    urban_rural_df['UrbanPercentage'] = urban_rural_df['area2'] * round((100 / sum(urban_rural_df['area2'])), 6)
    st.write("Let's try to understand the demographics of the urban and the rural population")
    urban_rural_df

    st.markdown("The following pattern is seen in both urban and rural areas:")
    st.markdown("- ~ 58-68% of the houses are in Target 4 (non-vulnerable)")
    st.markdown("- ~ 10-15% of the houses are in Target 2 (moderate)")
    st.markdown("- ~ 13-18% of the houses are in Target 3 (vulnerable)")
    st.markdown("- ~ 6-10% of the houses are in Target 1 (extreme)")
    st.write("""
           ***
           """)
    st.write("### **Region**")

    region_df = df_train_heads.groupby('Target').sum()[['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']]
    region_df
    st.markdown("lugar1	=1 region Central")
    st.markdown("lugar2	=1 region Chorotega")
    st.markdown("lugar3	=1 region PacÃƒÆ’Ã‚Â­fico central")
    st.markdown("lugar4	=1 region Brunca")
    st.markdown("lugar5	=1 region Huetar AtlÃƒÆ’Ã‚Â¡ntica")
    st.markdown("lugar6	=1 region Huetar Norte")
    st.pyplot(plot_dwelling_property(region_df))

    region_df.T
    st.markdown("- Central region has the maximum population")
    st.markdown("- A major portion (65%) of the poverty group 4 (non-vulnerable) people live in Central region")
    st.write("""
           ***
           """)
    st.write("### **Monthly Rent -**")

    st.write(round(((all_data.shape[0] - sum(all_data['v2a1'].value_counts())) / all_data.shape[0]) * 100, 2))
    st.markdown("A large percentage of v2a1 (the monthly rent column) is empty. We will analyse this after we input "
                "the missing values in the next section.")
    st.write("""
           ***
           """)
    st.write('### **Education -escolari**')

    sns.boxplot(x='Target', y='escolari', data=all_data);
    st.pyplot()
    st.markdown("### Conclusion:-")
    st.markdown("These features do not convey any useful information about the Target variable:")
    st.markdown("- 'sanitario1', 'sanitario6'")
    st.markdown("- 'elimbasu4', 'elimbasu5', 'elimbasu6'")
    st.markdown("- 'techozinc', 'techoentrepiso', 'techocane', 'techootro'")
    st.markdown("- 'abastaguadentro', 'abastaguafuera', 'abastaguano'")
    st.markdown("- 'public', 'planpri', 'noelec', 'coopele' Numerical or")
    all_data.drop(
        columns=['sanitario1', 'sanitario6', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'techozinc', 'techoentrepiso',
                 'techocane', 'techootro', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public',
                 'planpri', 'noelec', 'coopele'], inplace=True)

    st.markdown("Okay, so education affects the poverty label to some extent. Or, maybe poverty label affects one's "
                "ability to get education.")

    st.write("""
           ***
           """)
    st.write("### **Numerical or Categorical?**")

    num_features = all_data._get_numeric_data().columns
    num_features_length = len(num_features)

    categ_features = pd.Index(list(set(all_data.columns) - set(num_features)))
    categ_features_length = len(categ_features)

    st.write("Number of numerical features: ", num_features_length)
    st.write("Number of categorical features: ", categ_features_length)

    labels = ['numeric', 'categorical']
    colors = ['y', 'r']
    plt.figure(figsize=(8, 8))
    plt.pie([num_features_length, categ_features_length],
            labels=labels,
            autopct='%1.1f%%',
            shadow=True,
            colors=colors)
    st.pyplot()
    all_data[categ_features].head()

    _, axarr = plt.subplots(nrows=1, ncols=3, sharey='row', figsize=(12, 6))

    for idx, feature in enumerate(['dependency', 'edjefe', 'edjefa']):
        sns.countplot(x=feature, data=all_data[all_data[feature].isin(['yes', 'no'])], ax=axarr[idx])
    st.pyplot()
    st.write("A look at this showed that there is a glitch with dependency, edjefe and edjefa. In all of these cases,"
             ", 'yes' implies 1 and 'no' implies 0. So, let's fix that..")
    yes_no_map = {'no': 0, 'yes': 1}

    all_data['dependency'] = all_data['dependency'].replace(yes_no_map).astype(np.float32)
    all_data['edjefe'] = all_data['edjefe'].replace(yes_no_map).astype(np.float32)
    all_data['edjefa'] = all_data['edjefa'].replace(yes_no_map).astype(np.float32)

    st.write("""
           ***
           """)
    st.write("### **Numerical features that are binary**")

    num_binary_features = []

    for feature in all_data.columns:
        if sorted(df_train[feature].unique()) in [[0, 1], [0], [1]]:
            num_binary_features.append(feature)

    st.write("Total number of binary-numerical features: ", len(num_binary_features))
    st.write("Binary-numerical features: ")
    num_binary_features
    st.write("""
           ***
           """)
    st.write("### **Non-Binary Features**")

    num_non_binary_features = [feature for feature in all_data.columns if feature not in num_binary_features]
    st.write("Total number of non-binary-numerical features: ", len(num_non_binary_features))
    st.write("Non-binary numerical features: ")

    num_non_binary_features_dict = {feature: len(all_data[feature].unique()) for feature in num_non_binary_features}

    num_non_binary_features_sorted = sorted(num_non_binary_features_dict,
                                            key=lambda feature: num_non_binary_features_dict[feature], reverse=True)

    num_non_binary_features_len_sorted = [num_non_binary_features_dict[feature] for feature in
                                          num_non_binary_features_sorted]

    plt.figure(figsize=(16, 16))
    barplot_with_anotate(num_non_binary_features_sorted, num_non_binary_features_len_sorted)
    plt.ylabel("No. of unique values")
    plt.xlabel("Non-binary numerical features")
    st.pyplot()

    st.markdown("Out of these 39 features, the following are continuous in nature:")
    st.markdown("- v2al")
    st.markdown("- meaneduc")
    st.markdown("- SQBmeaned")
    st.markdown("- dependency")
    st.markdown("- SQBdependency")
    st.markdown("All the other features are discrete in nature.")
    st.write("""
           ***
           """)
    st.write("### **Binary features**")

    st.write(all_data[num_binary_features].describe())
    st.write("""
           ***
           """)
    st.write("### **Non-binary continuous features:**")

    num_conti_features = pd.Index(['v2a1', 'meaneduc', 'dependency', 'SQBmeaned', 'SQBdependency'])
    st.write(all_data[num_conti_features].describe())
    st.write("""
           ***
           """)
    st.write("### **Non-binary discrete features:**")

    num_discrete_features = pd.Index(
        [feature for feature in num_non_binary_features if feature not in num_conti_features])
    st.write(all_data[num_discrete_features].describe())

    st.write("""
           ***
           """)
    st.write("### **Missing Values Imputation**")


    def missing_features(data, column_set):
        incomplete_features = {feature: data.shape[0] - sum(data[feature].value_counts())
                               for feature in column_set
                               if not sum(data[feature].value_counts()) == data.shape[0]}
        incomplete_features_sorted = sorted(incomplete_features, key=lambda feature: incomplete_features[feature],
                                            reverse=True)
        incompleteness = [round((incomplete_features[feature] / data.shape[0]) * 100, 2) for feature in
                          incomplete_features_sorted]
        plt.figure(figsize=(12, 6))
        barplot_with_anotate(incomplete_features_sorted, incompleteness)
        plt.ylabel("Percentage (%) of values that are missing")
        st.pyplot()
        for feature, percentage in zip(incomplete_features_sorted, incompleteness):
            st.write("Feature:", feature)
            st.write("No. of NaNs:", incomplete_features[feature], "(", percentage, ")")


    missing_features(all_data, all_data.columns)

    st.markdown("- rez_esc (Years behind in school): NaN implies that the person does not remember. Considering that "
                "along with the large percentage of NaN values, we are better off dropping that column.")
    st.markdown("- meaneduc and SQBmeaned: With the average of the columns.")
    st.write("""
           ***
           """)
    st.write("### **v2a1:**")

    st.write(all_data[['v2a1', 'tipovivi3']][all_data['tipovivi3'] == 0][all_data['v2a1'].isnull()].shape)
    st.markdown("We see all those entries where v2a1 is NaN also have tipovivi3 as 0, which implies that all those "
                "houses are not rented.")
    st.markdown("Hence, we should fill the missing values of v2a1 with 0.")
    # handling v2a1
    st.write(all_data.loc[:, 'v2a1'].fillna(0, inplace=True))
    st.write("""
           ***
           """)
    st.write("### **v18q1:**")

    st.write(all_data[['v18q1', 'v18q']][all_data['v18q'] == 0][all_data['v18q1'].isnull()].shape)
    st.markdown("We see that v18q1 is NaN only for those entries which have v18q == 0, Thus, v18q1 is missing only "
                "when the house does not have a tablet. "
                "houses are not rented.")
    st.markdown("Hence, we should fill the missing values of v18q1 with 0.")
    # handling v18q1
    all_data.loc[:, 'v18q1'].fillna(0, inplace=True)
    st.write("""
           ***
           """)
    st.write("### **meaneduc and SQBmeaned:**")

    # handling meaneduc and SQBmeaned
    all_data.loc[:, 'meaneduc'].fillna(all_data['meaneduc'].mean(), inplace=True)
    all_data.loc[:, 'SQBmeaned'].fillna(all_data['SQBmeaned'].mean(), inplace=True)

    st.write("""
           ***
           """)
    st.write("### **rez_esc:**")
    st.write("Drop it")
    all_data.drop(columns=['rez_esc'], inplace=True)

    st.write("""
           ***
           """)
    st.write("### **Convey dummy to original:**")

    all_data['WallQual'] = all_data['epared1'] + 2 * all_data['epared2'] + 3 * all_data['epared3']
    all_data['RoofQual'] = all_data['etecho1'] + 2 * all_data['etecho2'] + 3 * all_data['etecho3']
    all_data['FloorQual'] = all_data['eviv1'] + 2 * all_data['eviv2'] + 3 * all_data['eviv3']
    all_data['EducationLevel'] = all_data['instlevel1'] + 2 * all_data['instlevel2'] + 3 * all_data['instlevel3'] + 4 * \
                                 all_data['instlevel4'] + 5 * all_data['instlevel5'] + 6 * all_data['instlevel6'] + 7 * \
                                 all_data['instlevel7'] + 8 * all_data['instlevel8'] + 9 * all_data['instlevel9']

    all_data.drop(columns=['epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1',
                           'eviv2', 'eviv3', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5'
        , 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9'], inplace=True)

    st.write("""
           ***
           """)
    st.write("### **remove redundant features:**")
    st.markdown("- v18q can be generated by v18q1")
    st.markdown("- mobilephone can be generated by qmobilephone")
    redundant_features = ['r4t1', 'r4t2', 'tamhog', 'tamviv', 'hhsize', 'r4t3', 'v18q', 'mobilephone']
    all_data.drop(columns=redundant_features, inplace=True)

    st.write("""
           ***
           """)
    st.write("### **Create new household-wise features**")
    st.markdown("1. Hand-engineered features:")
    st.markdown("- Monthly rent per room - v2a1/rooms")
    st.markdown("- Monthly rent per adult - v2a1/hogar_adul")
    st.markdown("- No. of adults per room - hogar_adul/rooms")
    st.markdown("- No. of adults per bedroom - hogar_adul/bedrooms")
    st.markdown("2. Average of individual-level features per household")
    st.markdown("3. Minimum of individual-level features per household")
    st.markdown("4. Maximum of individual-level features per household")
    st.markdown("5. Sum of individual-level features per household")
    st.markdown("6. Standard deviation of individual-level features per household")
    all_data['RentPerRoom'] = all_data['v2a1'] / all_data['rooms']

    all_data['AdultsPerRoom'] = all_data['hogar_adul'] / all_data['rooms']

    all_data['AdultsPerBedroom'] = all_data['hogar_adul'] / all_data['bedrooms']

    # individual level boolean features

    ind_bool = ['dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4'
        , 'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4',
                'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10',
                'parentesco11',
                'parentesco12', 'EducationLevel']

    # individual level ordered features
    ind_ordered = ['escolari', 'age']

    f = lambda x: x.std(ddof=0)
    f.__name__ = 'std_0'
    ind_agg = all_data.groupby('idhogar')[ind_ordered + ind_bool].agg(['mean', 'max', 'min', 'sum', f])

    new_cols = []
    for col in ind_agg.columns.levels[0]:
        for stat in ind_agg.columns.levels[1]:
            new_cols.append(f'{col}-{stat}')

    ind_agg.columns = new_cols
    ind_agg.head()

    st.write("Original number of features:", all_data.shape[1])

    all_data = all_data.merge(ind_agg, on='idhogar', how='left')

    st.write("Number of features after merging transformed individual level features", all_data.shape[1])

    all_data.drop(columns=ind_bool + ind_ordered, inplace=True)

    st.write("Number of features after dropping the individual level features", all_data.shape[1])

    all_data.head()

    st.sidebar.write("""Model Training - KNN""")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X = all_data.drop(columns=['Target', 'idhogar']).copy()
    Y = all_data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

    knn = KNeighborsClassifier(n_neighbors=25)
    knn.fit(X_train, y_train)
    scores = knn.predict(X_test)
    accuracyknn = accuracy_score(scores, y_test)
    st.sidebar.write("Accuracy(KNN):", accuracyknn)

    st.sidebar.write("""2. Naive Bayes Classifier""")

    from sklearn.naive_bayes import BernoulliNB

    gnb = BernoulliNB()
    gnb.fit(X_train, y_train)
    scores = gnb.predict(X_test)
    accuracynb = accuracy_score(scores, y_test)
    st.sidebar.write("Accuracy(NB):", accuracynb)

    st.sidebar.write("""3. Support Vector machine""")

    from sklearn.svm import SVC

    classifier = SVC(kernel='rbf', gamma='auto', random_state=0)
    classifier.fit(X_train, y_train)
    accuracysvm = classifier.score(X_test, y_test)
    st.sidebar.write("Accuracy", accuracysvm)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, make_scorer
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score

    st.sidebar.write("""4. Random Forest Classifier""")
    model = RandomForestClassifier(n_estimators=100, random_state=10,
                                   n_jobs=-1)
    model.fit(X_train, y_train)
    accuracyrf = model.score(X_test, y_test)
    st.sidebar.write("Accuracy", accuracyrf)

    st.write("""
    ***
    """)

    st.write("According to the results it is clearly understood that **Random Forest Classifier** is best model and "
             "Naive "
             "Bayes, KNN and Support Vector Machine performed very bad.")
    sns.set_theme(style='whitegrid')
    sns.barplot(x=['KNN', 'NB', 'SVM', 'RF'],
                y=[accuracyknn * 100, accuracynb * 100, accuracysvm * 100, accuracyrf * 100])
    st.pyplot()
