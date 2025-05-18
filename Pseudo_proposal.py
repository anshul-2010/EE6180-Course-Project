# Pseudocode for training ResNet with RAAR

for epoch in range(num_epochs):
    for images, labels in dataloader:

        # Forward pass
        outputs = model(images)  # e.g., ResNet-18
        loss_ce = cross_entropy(outputs, labels)

        # Compute Grad-CAM for current batch
        attention_maps = compute_grad_cam(model, images, target_layer='layer4[1].conv2')

        # Normalize attention maps per image
        norm_attention = normalize_attention_maps(attention_maps)  # shape: (B, H, W)

        # ---- Component 1: Spatial Entropy Regularization ----
        entropy_loss = 0
        for attn in norm_attention:
            attn_flat = attn.view(-1) + epsilon  # to avoid log(0)
            entropy_loss += -torch.sum(attn_flat * torch.log(attn_flat))
        entropy_loss /= batch_size

        # ---- Component 2: Inter-Class Divergence (Optional) ----
        # Split into real and fake
        real_attention = norm_attention[labels == REAL]
        fake_attention = norm_attention[labels == FAKE]
        if len(real_attention) > 0 and len(fake_attention) > 0:
            avg_real_attn = torch.mean(real_attention, dim=0)
            avg_fake_attn = torch.mean(fake_attention, dim=0)
            kl_div = torch.sum(avg_fake_attn * torch.log((avg_fake_attn + epsilon) / (avg_real_attn + epsilon)))
        else:
            kl_div = 0.0

        # ---- Component 3: Saliency-Aware Dropout (Patch Occlusion) ----
        images_aug = apply_saliency_aware_dropout(images, norm_attention, p=0.3)

        # Forward pass again (optional) with occluded images to enforce robustness
        outputs_aug = model(images_aug)
        loss_aug = cross_entropy(outputs_aug, labels)

        # Combine all loss components
        total_loss = loss_ce + lambda1 * entropy_loss + lambda2 * kl_div + lambda3 * loss_aug

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
