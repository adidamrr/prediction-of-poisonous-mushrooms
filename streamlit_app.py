import httpx
import os
import streamlit as st


st.set_page_config(
    page_title="Скрининг риска токсичности гриба",
    layout="centered",
)


SERVICE_URL = os.getenv("SERVICE_URL", "http://127.0.0.1:8000")

COLOR_OPTIONS = {
    "n": "Коричневый",
    "b": "Светло-бежевый",
    "g": "Серый",
    "w": "Белый",
    "y": "Желтый",
    "r": "Зеленоватый",
    "p": "Розовый",
    "u": "Фиолетовый",
    "e": "Красный",
    "o": "Оранжевый",
    "k": "Черный",
    "h": "Шоколадный",
    "l": "Голубоватый",
    "a": "Охристый",
    "f": "Палевый",
    "d": "Темный",
    "s": "Песочный",
    "t": "Светло-коричневый",
    "z": "Смешанный",
}

CAP_SHAPE_OPTIONS = {
    "b": "Колокольчатая",
    "c": "Коническая",
    "f": "Плоская",
    "o": "Округлая",
    "p": "Широкая",
    "s": "Вдавленная",
    "x": "Выпуклая",
}

BOOL_OPTIONS = {
    "t": "Да",
    "f": "Нет",
}

SEASON_OPTIONS = {
    "a": "Весна",
    "s": "Лето",
    "u": "Осень",
    "w": "Зима",
}

GILL_COLOR_OPTIONS = COLOR_OPTIONS
HAS_RING_OPTIONS = {
    "t": "Есть кольцо",
    "f": "Нет кольца",
}


def select_optional(label, options, key):
    values = ["Не указывать"] + list(options.keys())

    def render(value):
        if value == "Не указывать":
            return value
        return f"{options[value]} ({value})"

    result = st.selectbox(label, values, key=key, format_func=render)
    if result == "Не указывать":
        return None
    return result


def select_required(label, options, key):
    values = list(options.keys())

    def render(value):
        return f"{options[value]} ({value})"

    return st.selectbox(label, values, key=key, format_func=render)


def request_prediction(payload):
    response = httpx.post(f"{SERVICE_URL}/predict", json=payload, timeout=10.0)
    response.raise_for_status()
    return response.json()


def request_feedback(prediction_id, feedback_value):
    response = httpx.post(
        f"{SERVICE_URL}/feedback",
        json={
            "prediction_id": prediction_id,
            "feedback_value": feedback_value,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


def should_show_probability(score, threshold):
    return abs(score - threshold) <= threshold * 0.1


def get_risk_level(score, threshold):
    if score <= threshold:
        return None
    return (score - threshold) / (1 - threshold)


st.title("Скрининг риска токсичности гриба")
st.write(
    "Введите наблюдаемые признаки гриба. Сервис оценит риск токсичности и вернет предупреждение."
)

with st.form("prediction_form"):
    st.subheader("Обязательные признаки")

    col1, col2 = st.columns(2)
    with col1:
        cap_diameter = st.number_input("Диаметр шляпки", min_value=0.0, value=10.0, step=0.5)
    with col2:
        stem_width = st.number_input("Ширина ножки", min_value=0.0, value=2.0, step=0.5)

    col3, col4 = st.columns(2)
    with col3:
        stem_height = st.number_input("Высота ножки", min_value=0.0, value=8.0, step=0.5)
    with col4:
        stem_color = select_required("Цвет ножки", COLOR_OPTIONS, "stem_color")

    col5, col6 = st.columns(2)
    with col5:
        cap_color = select_required("Цвет шляпки", COLOR_OPTIONS, "cap_color")
    with col6:
        cap_shape = select_required("Форма шляпки", CAP_SHAPE_OPTIONS, "cap_shape")

    st.subheader("Дополнительные признаки")

    col3, col4 = st.columns(2)
    with col3:
        gill_color = select_optional("Цвет пластинок", GILL_COLOR_OPTIONS, "gill_color")
        has_ring = select_optional("Наличие кольца", HAS_RING_OPTIONS, "has_ring")
    with col4:
        bruises = select_optional("Изменение цвета / кровоподтеки", BOOL_OPTIONS, "bruises")
        season = select_optional("Сезон", SEASON_OPTIONS, "season")

    submitted = st.form_submit_button("Оценить риск", use_container_width=True)


if submitted:
    payload = {
        "cap-diameter": cap_diameter,
        "stem-height": stem_height,
        "stem-width": stem_width,
        "cap-color": cap_color,
        "stem-color": stem_color,
        "cap-shape": cap_shape,
        "does-bruise-or-bleed": bruises,
        "gill-color": gill_color,
        "season": season,
        "has-ring": has_ring,
    }
    payload = {key: value for key, value in payload.items() if value is not None}

    try:
        result = request_prediction(payload)
    except Exception as exc:
        st.error("Не удалось получить ответ от сервиса")
        st.caption(str(exc))
    else:
        st.session_state["prediction_result"] = result
        st.session_state["feedback_sent"] = False

result = st.session_state.get("prediction_result")
feedback_sent = st.session_state.get("feedback_sent", False)

if result is not None:
    decision = result["decision"]

    if decision == "unsafe":
        st.error(result["message"])
    elif decision == "review":
        st.warning(result["message"])
    else:
        st.success(result["message"])

    metric_col1, metric_col2 = st.columns(2)
    edible_label = "несъедобный" if result["label"] == "p" else "съедобный"
    with metric_col1:
        st.metric("Класс модели", edible_label)
    with metric_col2:
        risk_level = get_risk_level(result["prob_poisonous"], result["threshold"])
        if risk_level is not None:
            st.metric("Уровень риска", f"{risk_level:.1%}")
        elif should_show_probability(result["prob_poisonous"], result["threshold"]):
            st.metric("Скор модели", f"{result['prob_poisonous']:.1%}")

    if result["is_borderline"]:
        st.info("Пограничный результат. Помогите улучшить систему.")

        if feedback_sent:
            st.success("Спасибо, отзыв сохранён.")
        else:
            fb_col1, fb_col2 = st.columns(2)
            with fb_col1:
                if st.button("Скорее верно", use_container_width=True):
                    try:
                        request_feedback(result["prediction_id"], "correct")
                    except Exception as exc:
                        st.error("Не удалось сохранить отзыв")
                        st.caption(str(exc))
                    else:
                        st.session_state["feedback_sent"] = True
                        st.rerun()
            with fb_col2:
                if st.button("Скорее неверно", use_container_width=True):
                    try:
                        request_feedback(result["prediction_id"], "incorrect")
                    except Exception as exc:
                        st.error("Не удалось сохранить отзыв")
                        st.caption(str(exc))
                    else:
                        st.session_state["feedback_sent"] = True
                        st.rerun()
