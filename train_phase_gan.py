def train_phase_gan(device,
                    dataloader,
                    generator,
                    discriminator,
                    adversarial_loss,
                    l1_loss,
                    optimizer_G,
                    optimizer_D,
                    epochs,
                    lambda_l1):

    import torch
    real_label = 1.0
    fake_label = 0.0
    generator.train()
    discriminator.train()
    history = {
        "gen": [],
        "disc": []
    }

    for epoch in range(epochs):
        running_gen = 0.0
        running_disc = 0.0
        num_batches = 0
        for X_db, X_phase, Y_phase, _, _ in dataloader:
            X = torch.cat((X_db, X_phase), dim=1).to(device)
            X_db, X_phase = X_db.to(device), X_phase.to(device)
            Y_phase = Y_phase.to(device)

            # -------------------
            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            # Real pair
            real_pair = torch.cat((X, Y_phase), dim=1)
            real_output = discriminator(real_pair)
            real_loss = adversarial_loss(real_output, torch.ones_like(
                real_output, device=device) * real_label)

            # Fake pair
            Y_hat = generator(X)
            fake_pair = torch.cat((X, Y_hat), dim=1)
            # Detach to avoid updating G
            fake_output = discriminator(fake_pair.detach())
            fake_loss = adversarial_loss(fake_output, torch.zeros_like(
                fake_output, device=device) * fake_label)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # -------------------
            # Train Generator
            # -------------------
            optimizer_G.zero_grad()

            # Adversarial loss
            fake_output = discriminator(fake_pair)
            g_adv_loss = adversarial_loss(fake_output, torch.ones_like(
                fake_output, device=device) * real_label)

            # L1 loss for reconstruction
            g_l1_loss = l1_loss(Y_hat, Y_phase) * lambda_l1

            # Total generator loss
            g_loss = g_adv_loss + g_l1_loss
            g_loss.backward()
            optimizer_G.step()

            running_gen += g_loss.item()
            running_disc += d_loss.item()
            num_batches += 1
            if num_batches % 200 == 0:
                print(f" - B: {num_batches}\t D: {d_loss.item():.6f}\t G: {g_loss.item():.6f}\t Gadv: {g_adv_loss.item():.6f}")

        running_gen /= num_batches
        running_disc /= num_batches
        history["gen"].append(running_gen)
        history["disc"].append(running_disc)
        print(
            f"E: {epoch + 1:03d}/{epochs}\t D: {running_disc:.6f}\t G: {running_gen:.6f}")

    print("Training Complete!")
    return generator, history
