
from utils import scraping, PreProcessing

########################################################### Variables ###############################################################################################################################################################################################################################

bad_urls = ["https://circulareconomy.europa.eu/platform/","https://europa.eu/youreurope/business/finance-funding/getting-funding/access-finance/", 
            "https://ec.europa.eu/regional_policy/en/newsroom/coronavirus-response/react-eu", "https://ec.europa.eu/environment/europeangreencapital/index_en.htm", 
            "https://app.euplf.eu/#/","https://ec.europa.eu/eurostat/web/products-datasets/-/tour_occ_arnat", "https://ec.europa.eu/eurostat/web/products-datasets/-/tour_occ_arnraw", 
            "https://transport.ec.europa.eu/transport-themes/clean-transport-urban-transport/sumi_en", "https://ec.europa.eu/eurostat/databrowser/bookmark/bc5378b9-b6cb-468a-a4ac-055bb4d6f014?lang=en", 
            "https://ec.europa.eu/eurostat/databrowser/bookmark/20ee839b-c731-47b6-917e-01f8d37cbdca?lang=en", "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-search;callCode=null;freeTextSearchKeyword=;matchWholeText=true;typeCodes=0,1,2;statusCodes=31094501,31094502,31094503;programmePeriod=null;programCcm2Id=null;programDivisionCode=null;focusAreaCode=null;destination=null;mission=null;geographicalZonesCode=null;programmeDivisionProspect=null;startDateLte=null;startDateGte=null;crossCuttingPriorityCode=null;cpvCode=null;performanceOfDelivery=null;sortQuery=sortStatus;orderBy=asc;onlyTenders=false;topicListKey=topicSearchTablePageState", 
            "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/projects-results", "https://eudatasharing.eu/", 
            "https://ec.europa.eu/info/policies/consumers/consumer-protection-policy/green-consumption-pledge-initiative_en", 
            "https://ec.europa.eu/info/research-and-innovation/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe/missions-horizon-europe/healthy-oceans-seas-coastal-and-inland-waters_en", 
            "https://ec.europa.eu/info/research-and-innovation/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe/missions-horizon-europe/climate-neutral-and-smart-cities_en", 
            "https://europa.eu/eurobarometer/surveys/detail/2283", "http://europa.eu.int/citizensrights/signpost/about/index_en.htm#note1#note1"]

new_url = ["https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/internationalisation-tourism-businesses/marketing-your-tourism-company-internationally_en",
            "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/internationalisation-tourism-businesses/transnational-business-cooperation_en", "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/internationalisation-tourism-businesses/international-market-selection_en",
            "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/getting-know-potential-clients/traditional-european-markets_en", "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/getting-know-potential-clients/usa-main-traditional-international-market_en",
            "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/getting-know-potential-clients/emerging-markets_en", "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/running-your-business/hosting-20-events_en",
            "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/running-your-business/public-relations_en", "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/running-your-business/customer-services-europe_en",
            "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/financing-your-business/tourism-related-taxes-across-eu_en", "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/understanding-legislation/regulation-tourism-activity-europe_en", 
            "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/understanding-legislation/european-tourism-legislation_en", "https://single-market-economy.ec.europa.eu/sectors/tourism/eu-funding-and-businesses/business-portal/accessibility_en", 
            "https://www.oneplanetnetwork.org/programmes/sustainable-tourism", "https://www.switch-asia.eu/site/assets/files/3446/cambodia_roadmap_-_july_2022-1.pdf", "https://www.oneplanetnetwork.org/sites/default/files/from-crm/10YFP%2520annual%2520report.pdf",
            "https://www.oneplanetnetwork.org/sites/default/files/from-crm/Issue%252002_July%25202023.pdf", "https://www.eif.org/what_we_do/guarantees/case-studies/sme-initiative-spain-brujula.htm?lang=-en", "https://www.eif.org/what_we_do/guarantees/case-studies/efsi_cosme_zegluga_srodladowa_tadeusz_prokop_poland.htm?lang=-en",
            "https://www.eif.org/what_we_do/guarantees/case-studies/elite-travel-albania.htm?lang=-en", "https://www.eif.org/what_we_do/equity/Case_studies/questo-romania.htm?lang=-en",
            "https://nexttourismgeneration.eu/wp-content/uploads/2020/07/Survey-report-Bulgaria-NTG.pdf", "https://nexttourismgeneration.eu/wp-content/uploads/2019/03/NTG_Desk_Research_Summary_January_2019.pdf",
            "https://nexttourismgeneration.eu/wp-content/uploads/2020/07/Interview-report-Bulgaria-NTG.pdf", "https://www.cedefop.europa.eu/en/data-insights/skills-developments-and-trends-tourism-sector#_employment_in_tourism",
            "https://www.cedefop.europa.eu/en/data-insights/skills-developments-and-trends-tourism-sector#_job_and_skills_demand_in_tourism", "https://www.cedefop.europa.eu/en/data-insights/skills-developments-and-trends-tourism-sector#_covid19_pandemic_and_future_of_tourism",
            "https://www.cedefop.europa.eu/en/data-insights/skills-developments-and-trends-tourism-sector#_how_can_skill_needs_in_tourism_be_met", "https://new-european-bauhaus.europa.eu/about/about-initiative_en", 
            "https://food.ec.europa.eu/horizontal-topics/farm-fork-strategy/legislative-framework_en", "https://ec.europa.eu/eurostat/documents/10186/10693286/GFS-guidance-note-statistical-recording-recovery-resilience-facility.pdf/4117dec2-7840-a80d-7cb8-6d4f48c90a5a?t=1633505104650",
            "https://commission.europa.eu/system/files/2021-05/swd-annual-single-market-report-2021_en.pdf", "https://ec.europa.eu/regional_policy/sources/policy/themes/outermost-regions/covid19_or_study_en.pdf", "https://etc-corporate.org/uploads/2021/11/ETC-Quarterly-Report-Q3-2021_Public.pdf",
            "https://ec.europa.eu/info/funding-tenders/opportunities/docs/2021-2027/horizon/wp-call/2023-2024/wp-9-food-bioeconomy-natural-resources-agriculture-and-environment_horizon-2023-2024_en.pdf", "https://ec.europa.eu/info/funding-tenders/opportunities/docs/2021-2027/horizon/wp-call/2023-2024/wp-5-culture-creativity-and-inclusive-society_horizon-2023-2024_en.pdf",
            "https://transport.ec.europa.eu/system/files/2021-04/2021-mobility-strategy-and-action-plan.pdf", "https://ec.europa.eu/eurostat/documents/7870049/10293066/KS-FT-19-007-EN-N.pdf/f9cdc4cc-882b-5e29-03b1-f2cee82ec59d?t=1575909526000",
            "https://commission.europa.eu/system/files/2019-06/com_2019_270_f1_report_from_commission_en.pdf", "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32022R1925", "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/how-to-participate/org-details/999999999/project/101083920/program/43152860/details",
            "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/how-to-participate/org-details/999999999/project/101084007/program/43152860/details", "https://digital-strategy.ec.europa.eu/en/library/data-act-proposal-regulation-harmonised-rules-fair-access-and-use-data"
            ]

##########################################################################################################################################################################################################################################################################################

if __name__=="__main__":

    file = "TransitionPathwayForTourism.pdf" # defining the file from which urls need to be scraped
    urls = scraping.addingURLs(scraping.scrapingURLs(file), bad_urls, new_url) # Scraping urls from file

    content = scraping.PDFscraping("PDF resources", scraping.webScraping(urls)) # Scraping the content of relevant files
    
    n = 0
    cleaned_content = []
    for text in content:
        cleaned_content.append(PreProcessing.PreProcessing(text, n))



