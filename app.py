import streamlit as st
import pandas as pd
import base64
import io
import tldextract

from http.client import HTTPSConnection
from base64 import b64encode
from json import loads, dumps

# -------------
# Constants
# -------------
LANGUAGES = ["Afrikaans","Albanian","Amharic","Arabic","Armenian","Azerbaijani","Basque","Belarusian","Bengali","Bosnian","Bulgarian","Catalan","Cebuano","Chinese (Simplified)","Chinese (Traditional)","Corsican","Croatian","Czech","Danish","Dutch","English","Esperanto","Estonian","Finnish","French","Frisian","Galician","Georgian","German","Greek","Gujarati","Haitian Creole","Hausa","Hawaiian","Hebrew","Hindi","Hmong","Hungarian","Icelandic","Igbo","Indonesian","Irish","Italian","Japanese","Javanese","Kannada","Kazakh","Khmer","Kinyarwanda","Korean","Kurdish","Kyrgyz","Lao","Latvian","Lithuanian","Luxembourgish","Macedonian","Malagasy","Malay","Malayalam","Maltese","Maori","Marathi","Mongolian","Myanmar (Burmese)","Nepali","Norwegian","Nyanja (Chichewa)","Odia (Oriya)","Pashto","Persian","Polish","Portuguese (Portugal","Punjabi","Romanian","Russian","Samoan","Scots Gaelic","Serbian","Sesotho","Shona","Sindhi","Sinhala (Sinhalese)","Slovak","Slovenian","Somali","Spanish","Sundanese","Swahili","Swedish","Tagalog (Filipino)","Tajik","Tamil","Tatar","Telugu","Thai","Turkish","Turkmen","Ukrainian","Urdu","Uyghur","Uzbek","Vietnamese","Welsh","Xhosa","Yiddish","Yoruba","Zulu"]
COUNTRIES = ["Afghanistan", "Albania", "Antarctica", "Algeria", "American Samoa", "Andorra", "Angola", "Antigua and Barbuda", "Azerbaijan", "Argentina", "Australia", "Austria", "The Bahamas", "Bahrain", "Bangladesh", "Armenia", "Barbados", "Belgium", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Belize", "Solomon Islands", "Brunei", "Bulgaria", "Myanmar (Burma)", "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Central African Republic", "Sri Lanka", "Chad", "Chile", "China", "Christmas Island", "Cocos (Keeling) Islands", "Colombia", "Comoros", "Republic of the Congo", "Democratic Republic of the Congo", "Cook Islands", "Costa Rica", "Croatia", "Cyprus", "Czechia", "Benin", "Denmark", "Dominica", "Dominican Republic", "Ecuador", "El Salvador", "Equatorial Guinea", "Ethiopia", "Eritrea", "Estonia", "South Georgia and the South Sandwich Islands", "Fiji", "Finland", "France", "French Polynesia", "French Southern and Antarctic Lands", "Djibouti", "Gabon", "Georgia", "The Gambia", "Germany", "Ghana", "Kiribati", "Greece", "Grenada", "Guam", "Guatemala", "Guinea", "Guyana", "Haiti", "Heard Island and McDonald Islands", "Vatican City", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Kazakhstan", "Jordan", "Kenya", "South Korea", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Lesotho", "Latvia", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Mauritania", "Mauritius", "Mexico", "Monaco", "Mongolia", "Moldova", "Montenegro", "Morocco", "Mozambique", "Oman", "Namibia", "Nauru", "Nepal", "Netherlands", "Curacao", "Sint Maarten", "Caribbean Netherlands", "New Caledonia", "Vanuatu", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "Norfolk Island", "Norway", "Northern Mariana Islands", "United States Minor Outlying Islands", "Federated States of Micronesia", "Marshall Islands", "Palau", "Pakistan", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Pitcairn Islands", "Poland", "Portugal", "Guinea-Bissau", "Timor-Leste", "Qatar", "Romania", "Rwanda", "Saint Helena, Ascension and Tristan da Cunha", "Saint Kitts and Nevis", "Saint Lucia", "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Vietnam", "Slovenia", "Somalia", "South Africa", "Zimbabwe", "Spain", "Suriname", "Eswatini", "Sweden", "Switzerland", "Tajikistan", "Thailand", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago", "United Arab Emirates", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "North Macedonia", "Egypt", "United Kingdom", "Guernsey", "Jersey", "Tanzania", "United States", "Burkina Faso", "Uruguay", "Uzbekistan", "Venezuela", "Wallis and Futuna", "Samoa", "Yemen", "Zambia"]
DEVICES = ["mobile", "desktop"]

# -------------
# Variables
# -------------

preferred_countries = ["Germany", "Austria", "Switzerland", "United Kingdom", "United States", "France", "Italy", "Netherlands"]
preferred_languages = ["German", "English", "French", "Italian", "Dutch"]

# -------------
# Classes
# -------------

class RestClient:
    domain = "api.dataforseo.com"

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def request(self, path, method, data=None):
        connection = HTTPSConnection(self.domain)
        try:
            base64_bytes = b64encode(
                ("%s:%s" % (self.username, self.password)).encode("ascii")
                ).decode("ascii")
            headers = {'Authorization' : 'Basic %s' %  base64_bytes, 'Content-Encoding' : 'gzip'}
            connection.request(method, path, headers=headers, body=data)
            response = connection.getresponse()
            return loads(response.read().decode())
        finally:
            connection.close()

    def get(self, path):
        return self.request(path, 'GET')

    def post(self, path, data):
        if isinstance(data, str):
            data_str = data
        else:
            data_str = dumps(data)
        return self.request(path, 'POST', data_str)


# -------------
# Streamlit App Configuration
# -------------

st.set_page_config(
    page_title="Keyword Tool",
    page_icon=":mag:",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/',
        'About': "This is an app for keyword analysis"
    }
)


def setup_streamlit():
    st.image("https://www.claneo.com/wp-content/uploads/Element-4.svg", width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption("👋 Developed by [Kevin](https://www.linkedin.com/in/kirchhoff-kevin/)") 
    st.title("Get Keyword Data")
    st.divider()




def custom_sort(all_items, preferred_items):
  sorted_items = preferred_items + ["_____________"] + [item for item in all_items if item not in preferred_items]
  return sorted_items

def get_search_volume(df_chunk, client, country_name, language_name, device, check_seasonality):
    sv_results = []
    sv_errors = []
    sv_post_data = [{
        "location_name": country_name,
        "language_name": language_name,
        "keywords": df_chunk[df_chunk.columns[0]].to_list(),
        "device": device
    }]
  
    try:
        sv_response = client.post("/v3/keywords_data/google_ads/search_volume/live", sv_post_data)
        if sv_response['status_code'] == 20000:
            for task in sv_response['tasks']:
                if task['result'] and len(task['result']) > 0:
                    for resultTaskInfo in task['result']:
                        if check_seasonality:
                            keyword = resultTaskInfo['keyword']
                            search_volume = resultTaskInfo['search_volume']
                            monthly_searches = resultTaskInfo['monthly_searches']
                            seasonality = any(
                                month['search_volume'] > search_volume for month in monthly_searches
                            )
                            max_sv = max(month['search_volume'] for month in monthly_searches)
                            months_max_sv = [
                                month['month'] for month in monthly_searches
                                if month['search_volume'] == max_sv
                            ]
                            sv_results.append([keyword, search_volume, seasonality, max_sv, months_max_sv])
                        else:
                            sv_results.append([resultTaskInfo['keyword'], resultTaskInfo['search_volume']])
                else:
                    sv_errors.append(task)
                    st.write(sv_errors)
        else:
            print(f"Error. Code: {sv_response['status_code']} Message: {sv_response['status_message']}")
    except Exception as e:
        sv_errors.append(str(e))
        print(sv_errors)
  
    if check_seasonality:
        columns = ['keyword', 'search_volume', 'seasonality', 'max_sv', 'months_max_sv']
    else:
        columns = ['keyword', 'search_volume']
  
    return pd.DataFrame(sv_results, columns=columns), sv_errors

def get_ranking_positions(client, df, country, language, device):
  results = []
  errors = []
  total_keywords = len(df)
  progress_text = st.empty()
  progress_bar = st.progress(0)
  
  for index, row in enumerate(df.itertuples(), 1):
      keyword = row.keyword
      progress_text.text(f"Retrieving SERP for keyword {index} of {total_keywords}: {keyword}")
      
      post_data = dict()
      post_data[len(post_data)] = dict(
          location_name=country,
          language_name=language,
          keyword=keyword,
          device=device
      )
      response = client.post("/v3/serp/google/organic/live/advanced", post_data)
      try:
          if response['status_code'] == 20000:
              for task in response['tasks']:
                  if task['result'] and len(task['result']) > 0:
                      for resultTaskInfo in task['result']:
                          results.append(resultTaskInfo)
                  else:
                      errors.append(task)
          else:
              print(f"Error. Code: {response['status_code']} Message: {response['status_message']}")
      except:
          errors.append(post_data)
          print(errors)
  
      progress_bar.progress(index / total_keywords)
  
  progress_text.empty()
  return results, errors

def process_serp_results(results, sv, domain, competitors):
  # Extract the domain name and TLD from the input
  domain_parts = tldextract.extract(domain)
  domain = f"{domain_parts.domain}.{domain_parts.suffix}"
  
  cleaned_competitors = []
  for competitor in competitors:
      competitor_parts = tldextract.extract(competitor)
      cleaned_competitor = f"{competitor_parts.domain}.{competitor_parts.suffix}"
      cleaned_competitors.append(cleaned_competitor)
  
  df_results = pd.DataFrame(results)
  organic_data = []

  total_iterations = len(df_results)

  report_data = []
  serp_data = []
  
  for j in range(total_iterations):
      keyword = df_results.loc[j, "keyword"]
      organic_result = 1
      own_ranking = None
      own_ranking_url = None
      competitor_rankings = {competitor: None for competitor in cleaned_competitors}
      
      for i in range(len(df_results.loc[j, "items"])):
          if df_results.loc[j, "items"][i]["type"] == "organic":
              result_domain_parts = tldextract.extract(df_results.loc[j, "items"][i]["domain"])
              result_domain = f"{result_domain_parts.domain}.{result_domain_parts.suffix}"
              domain_match = any(result_domain.lower() == competitor.lower() for competitor in [domain] + cleaned_competitors)
              
              serp_data.append({
                  'keyword': keyword,
                  'position': organic_result,
                  'position_with_serp-features': df_results.loc[j, "items"][i]["rank_absolute"],
                  'url': df_results.loc[j, "items"][i]["url"]
              })
              
              if domain_match:
                      organic_data.append({
                          'keyword': keyword,
                          'position_with_serp-features': df_results.loc[j, "items"][i]["rank_absolute"],
                          'position': organic_result,
                          'title': df_results.loc[j, "items"][i]["title"],
                          'domain': result_domain,
                          'url': df_results.loc[j, "items"][i]["url"]
                      })
                      
              if result_domain.lower() == domain.lower():
                      own_ranking = organic_result
                      own_ranking_url = df_results.loc[j, "items"][i]["url"]
              else:
                  for competitor in cleaned_competitors:
                      if result_domain.lower() == competitor.lower():
                              competitor_rankings[competitor] = organic_result
                              break
              
              organic_result += 1
      
      report_row = {'keyword': keyword, f'{domain}_ranking_url': own_ranking_url, f'{domain}_ranking': own_ranking}
      for competitor in competitors:
          report_row[f'{competitor}_ranking'] = competitor_rankings[competitor]
      report_data.append(report_row)
  
  report_df = pd.DataFrame(report_data)
  report_df = pd.merge(report_df, sv, on='keyword', how='left')
  report_columns = ['keyword', 'search_volume'] + [col for col in report_df.columns if col not in ['keyword', 'search_volume']]
  report_df = report_df.reindex(columns=report_columns)
  
  serp_df = pd.DataFrame(serp_data)
  serp_df = pd.merge(serp_df, sv, on='keyword', how='left')
  serp_columns = ['keyword', 'search_volume', 'position', 'position_with_serp-features', 'url']
  serp_df = serp_df.reindex(columns=serp_columns)
  
  
  return report_df, serp_df

def transpose_serp_results(serp_df):
  top10serp_df = serp_df[serp_df["position"] <= 10]
  top10serp_df.reset_index(drop=True, inplace=True)

  transposed_df = top10serp_df.pivot_table(index=['keyword', 'search_volume'], columns='position', values=['position_with_serp-features', 'url'], aggfunc='first')
  
  # Flatten the multi-level column index
  transposed_df.columns = [f'{col[1]}_{col[0]}' for col in transposed_df.columns]
  
  # Reset the index to make 'keyword' and 'search_volume' regular columns
  transposed_df.reset_index(inplace=True)
  
  # Reorder the columns
  column_order = ['keyword', 'search_volume']
  for i in range(1, 11):
      column_order.append(f'{i}_position_with_serp-features')
      column_order.append(f'{i}_url')
  transposed_df = transposed_df[column_order]
  
  return transposed_df
  

def clean_up_string(s):
  if not isinstance(s, str):
      return s # Return the original value if it's not a string

  # Remove the '@@' separators
  cleaned = s.replace('@@', '\n')
  # Split by newline to get individual elements
  elements = [elem.strip() for elem in cleaned.split('\n') if elem.strip()]
  return ' '.join(elements) # Joining the elements back into a single string



def download_excel_link(report, name):
  excel_buffer = io.BytesIO()
  report.to_excel(excel_buffer, index=False, sheet_name='Sheet1')
  excel_data = excel_buffer.getvalue()
  b64_excel = base64.b64encode(excel_data).decode()
  href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="{name}.xlsx">Download {name}</a>'
  st.markdown(href, unsafe_allow_html=True)

def show_dataframe(report):
  """
  Shows a preview of the first 100 rows of the report DataFrame in an expandable section.
  """
  with st.expander("Preview the First 100 Rows"):
      st.dataframe(report.head(100))
      
def chunk_dataframe(df, chunk_size=1000):
  chunks = []
  num_chunks = len(df) // chunk_size + 1
  for i in range(num_chunks):
      start = i * chunk_size
      end = (i + 1) * chunk_size
      chunk = df[start:end]
      chunks.append(chunk)
  return chunks


def main():
  setup_streamlit()
  username = st.text_input("DataforSEO Login", help="Get your login credentials here: https://app.dataforseo.com/api-dashboard")
  password = st.text_input('Please enter your DataforSEO API Password', type="password")
  client = RestClient(str(username), str(password))
  # Upload Excel file
  uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
  sorted_countries = custom_sort(COUNTRIES, preferred_countries)
  sorted_languages = custom_sort(LANGUAGES, preferred_languages)

  country = st.selectbox("Country", sorted_countries)
  language = st.selectbox("Language", sorted_languages)
  device = st.selectbox("Device", DEVICES)

  if uploaded_file is not None:
      # Read Excel file into a DataFrame
      df = pd.read_excel(uploaded_file)
      df.rename(columns={"Search term": "Keyword", "keyword": "Keyword", "query": "Keyword", "query": "Keyword", "Top queries": "Keyword", "queries": "Keyword", "Keywords": "Keyword","keywords": "Keyword", "Search terms report": "Keyword"}, inplace=True)
      # Check if the 'Keyword' column exists
      if 'Keyword' not in df.columns:
          st.error("Please make sure your Excel file contains a column named 'Keyword'!")
      else:
          # Input domain and competitors
          domain = st.text_input("Enter your domain")
          num_competitors = st.number_input("Enter the number of competitors", min_value=1, value=1, step=1)

          competitors = []
          for i in range(num_competitors):
              competitor = st.text_input(f"Enter competitor {i+1}")
              competitors.append(competitor)

          # Checkbox for including organic ranking position and people also search
          check_seasonality = st.checkbox("Check Seasonality")

          

          if st.button("Analyze"):
              # Get search volume
              chunks = chunk_dataframe(df)
              all_sv_results = pd.DataFrame()
              all_sv_errors = []

              for chunk in chunks:
                  sv_results, sv_errors = get_search_volume(chunk, client, country, language, device, check_seasonality)
                  all_sv_results = pd.concat([all_sv_results, sv_results])
                  all_sv_errors.extend(sv_errors)
              
              sv = pd.DataFrame(all_sv_results)

              # Get ranking positions
              results, errors = get_ranking_positions(client, sv, country, language, device)

              # Process SERP results
              report_df, serp_df = process_serp_results(results, sv, domain, competitors)
              
              # Display the results
              st.subheader("Report")
              show_dataframe(report_df)
              download_excel_link(report_df, "keyword_analysis_results")
              
              top10serp_df = transpose_serp_results(serp_df)

              st.subheader("Top 10 SERP Data")
              show_dataframe(top10serp_df)
              download_excel_link(top10serp_df, "top-10_serp_data")
