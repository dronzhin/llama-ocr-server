import streamlit as st
from modules.llama_prompt import llama11b

def llama_tab(uploaded_file):
    st.header("Llama")
    user_input = st.text_input("Введите текст:", key="llama_input")
    if user_input:
        st.session_state.llama_text = user_input
        st.write(f"Вы ввели: `{user_input}`")

        # Кнопка для обработки текста и вывода таблицы
        if st.button("Обработать текст (Llama)", key="llama_button"):
            try:
                prompt = 'Найти в тексте только показатели крови (indicator), их результат (result) и единицы измерения (measurement)'
                path_file = f'temp_{uploaded_file.name}'
                with open(path_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())  # Сохраняем файл на диск
                result = llama11b(message=prompt, path_image=path_file)
                st.session_state.df_llama = result  # Сохраняем результат в session_state
            except Exception as e:
                st.error(f"Ошибка при обработке текста: {e}")
    else:
        st.info("Введите текст в поле выше.")

     # Отображение и редактирование таблицы
    if 'df_llama' in st.session_state:
        st.subheader("Обработанный текст (Llama) в виде таблицы:")
        
        # Используем data_editor для редактирования
        edited_df_llama = st.data_editor(st.session_state.df_llama, 
                                    num_rows="dynamic", 
                                    key="edited_df_llama")

        # Кнопка для сохранения изменений (если нужно)
        if st.button("Сохранить изменения", key="save_llama"):
            st.session_state.df_llama = edited_df_llama
            st.success("Изменения сохранены!")

    # Кнопка сохранения в JSON
    if 'df_llama' in st.session_state:
        # Преобразуем DataFrame в JSON строку
        json_data = st.session_state.df_llama.to_json(orient="records", force_ascii=False, indent=4)

        # Кнопка для скачивания JSON
        st.download_button(
            label="Сохранить таблицу (Llama) в JSON",
            data=json_data,
            file_name="llama_table.json",
            mime="application/json"
    )  