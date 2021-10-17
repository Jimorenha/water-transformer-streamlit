mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"jimorenaaa@gmail.com\"\n\
" > ~/.streamlit/config.toml

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml