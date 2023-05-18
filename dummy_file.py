# model, class_to_idx = load_checkpoint('checkpoint_ic_d161.pth')

# def get_prediction(image_bytes):
#     try:
#         tensor = transform_image(image_bytes=image_bytes)
#         outputs = model.forward(tensor)
#     except Exception:
#         return 0, 'error'
#     _, y_hat = outputs.max(1)
#     predicted_idx = str(y_hat.item())
#     return class_to_idx[predicted_idx]
