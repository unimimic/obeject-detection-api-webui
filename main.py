import gradio as gr
from webui.module import create_project_directory, get_project_names, update_fields, save_project_settings, get_models, download_model, models
from webui.od import train, getTFRecord, export

def update_ui(project_name):
    if project_name:
        # 顯示與專案相關的設置項目
        return (gr.update(visible=True),  # dataset_format
                gr.update(visible=True),  # training_classes
                gr.update(visible=True),  # batch_size
                gr.update(visible=True),  # num_steps
                gr.update(visible=True),  # checkpoint_every_n
                gr.update(visible=True),  # reference_model
                gr.update(visible=True),  # task_name
                gr.update(visible=True),  # save_button
                gr.update(visible=True),  # download_button
                gr.update(visible=True))  # model_dropdown
    else:
        # 隱藏與專案相關的設置項目
        return (gr.update(visible=False),  # dataset_format
                gr.update(visible=False),  # training_classes
                gr.update(visible=False),  # batch_size
                gr.update(visible=False),  # num_steps
                gr.update(visible=False),  # checkpoint_every_n
                gr.update(visible=False),  # reference_model
                gr.update(visible=False),  # task_name
                gr.update(visible=False),  # save_button
                gr.update(visible=False),  # download_button
                gr.update(visible=False))  # model_dropdown

with gr.Blocks() as demo:
    with gr.Blocks():
        gr.Markdown("專案資訊")
        project_name = gr.Dropdown(get_project_names(), label="專案名稱")

        with gr.Accordion("New Project", open=False):
            new_project_name = gr.Textbox(label="名稱")
            create_button = gr.Button("新增")

        with gr.Row():
            dataset_format = gr.Dropdown(["xml", "json"], label="資料集格式")
            training_classes = gr.Textbox(label="訓練類別", lines=1, placeholder="Enter comma-separated strings...")
        
        with gr.Row():
            batch_size = gr.Number(value=1, maximum=30, minimum=1, step=1, label="Batch Size", interactive=True)
            num_steps = gr.Number(value=1, maximum=1000000, minimum=1, step=1, label="Steps")
            checkpoint_every_n = gr.Number(value=1, maximum=1000000, minimum=1, step=1, label="Checkpoint every n")

        reference_model = gr.Dropdown(get_models(), label="參考模型")

        with gr.Accordion("Download Model", open=False):
            model_dropdown = gr.Dropdown(models, label="模型名稱")
            download_button = gr.Button("下載模型")

        task_name = gr.Textbox(label="當次訓練任務名稱")
        save_button = gr.Button("存取以上資訊")


    output_text = gr.Textbox(label="輸出結果", interactive=False)
    with gr.Row():
        get_tfrecord_button = gr.Button("轉換資料")
        train_button = gr.Button("開始訓練")
        export_button = gr.Button("輸出模型")

    # 綁定create_button的事件處理函數
    export_button.click(
        fn=export, 
        inputs=[
            project_name,
            task_name
        ],
        outputs=output_text
    )


    train_button.click(
        fn=train, 
        inputs=[
            project_name,
            task_name,
            batch_size, 
            num_steps, 
            checkpoint_every_n, 
            reference_model
        ],
        outputs=output_text
    )

    get_tfrecord_button.click(
        fn=getTFRecord, 
        inputs=[project_name, dataset_format, task_name],
        outputs=output_text
    )

    create_button.click(
        fn=create_project_directory,
        inputs=new_project_name,
        outputs=output_text
    )

    project_name.change(
        fn=lambda pn: update_ui(pn),
        inputs=project_name,
        outputs=[
            dataset_format, 
            training_classes, 
            batch_size, 
            num_steps, 
            checkpoint_every_n, 
            reference_model,
            task_name
        ]
    )

    project_name.change(
        fn=update_fields,
        inputs=project_name,
        outputs=[
            dataset_format, 
            training_classes, 
            batch_size, 
            num_steps, 
            checkpoint_every_n, 
            reference_model,
            task_name
        ]
    )

    save_button.click(
        fn=save_project_settings, 
        inputs=[project_name, dataset_format, training_classes, batch_size, num_steps, checkpoint_every_n, reference_model, task_name],
        outputs=output_text
    )

    download_button.click(
        fn=download_model,
        inputs=model_dropdown,
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()